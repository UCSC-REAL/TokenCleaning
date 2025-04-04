#!/usr/bin/env python
# coding=utf-8
import sys
import argparse
import logging
import math
import os
import random
import datasets
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import torch
import torch.nn.functional as F

def compute_kl_div_score(ground_truth_logits, predicted_logits, reduction='none'):
    """
    计算两个模型的 KL 散度 (Kullback-Leibler Divergence)。

    参数:
        ground_truth_logits (torch.Tensor): 作为参考的模型 logits (通常是更稳定/更好的模型)，形状 (batch_size, seq_len, vocab_size)。
        predicted_logits (torch.Tensor): 目标模型 logits (需要评估的模型)，形状 (batch_size, seq_len, vocab_size)。
        reduction (str): KL 散度的 reduction 方法 ('batchmean', 'sum', 'mean', 'none')。

    返回:
        torch.Tensor: 计算得到的 KL 散度分数。
    """
    # 确保两个 logits 形状一致
    assert ground_truth_logits.shape == predicted_logits.shape, "logits 形状必须相同"

    # 计算 log-probabilities (log softmax)
    log_probs_gt = F.log_softmax(ground_truth_logits, dim=-1)  # Ground-truth 的 log 概率
    probs_pred = F.softmax(predicted_logits, dim=-1)  # 预测模型的普通概率分布

    # 计算 KL 散度
    kl_div = F.kl_div(log_probs_gt, probs_pred, reduction=reduction)
    kl_div = kl_div.sum(dim=-1)
    return kl_div

# cache_dir = '/tmp/huggingface/hub/'
# os.makedirs(cache_dir, exist_ok=True)

# cache_dir=None

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}.")

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--ref_model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log the training loss and learning rate every logging_steps steps.",
    )

    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        "--with_prompt_token", type=str2bool, default=False, help="whether to use prompt tokens in the selection process"
    )
    args = parser.parse_args()
    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, with_prompt_token, add_bos=False):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    
    # mask the prompt part for avoiding loss
    if not with_prompt_token:
        labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, with_prompt_token, add_bos=False):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            
            ### mask prompt loss
            if not with_prompt_token:
                labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def main():
    args = parse_args()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)


    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
    
    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        # logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            # cache_dir=cache_dir,

            
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None),
            # cache_dir=cache_dir

        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True  
            # )

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                quantization_config=bnb_config,
                device_map='auto',
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                device_map='auto',
                trust_remote_code=args.trust_remote_code,
                # low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
                # cache_dir=cache_dir,
            )
    else:
        print("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    ###############################################
    ############# reference_model #################
    ###############################################
    ref_config = AutoConfig.from_pretrained(
        args.ref_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.model_revision,
        token=os.getenv("HF_TOKEN", None)
        )
    if args.ref_model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            # bnb_config = BitsAndBytesConfig(
            #     load_in_8bit=True  
            # )

            reference_model = AutoModelForCausalLM.from_pretrained(
                args.ref_model_name_or_path,
                from_tf=bool(".ckpt" in args.ref_model_name_or_path),
                config=ref_config,
                quantization_config=bnb_config,
                device_map='auto',
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
            )
        else:
            reference_model = AutoModelForCausalLM.from_pretrained(
                args.ref_model_name_or_path,
                from_tf=bool(".ckpt" in args.ref_model_name_or_path),
                config=ref_config,
                device_map='auto',
                trust_remote_code=args.trust_remote_code,
                # low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision,
                token=os.getenv("HF_TOKEN", None),
                # cache_dir=cache_dir,
            )
            # reference_model.resize_token_embeddings(len(tokenizer))

    else:
        print("Training new model from scratch")
        reference_model = AutoModelForCausalLM.from_config(ref_config)
        # reference_model.resize_token_embeddings(len(tokenizer))





    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert args.add_bos, "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>'})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # gather deepspeed to get "real" embedding size    
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
    # resize does its own gather
    if len(tokenizer) > embedding_size:
        # pad to multiple for tensor cores.
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    # update embedding size after resizing for sum loss
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # set the tokenizer chat template to the tulu format
    # this makes evaluation/etc easier down the line.
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}" # noqa: E501

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "completion" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            with_prompt_token=args.with_prompt_token,
            add_bos=args.add_bos,
        )
    elif "messages" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            with_prompt_token=args.with_prompt_token,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
        
    raw_datasets = raw_datasets.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,  
        num_proc=args.preprocessing_num_workers,
        desc="Adding idx column",
    )
    
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["idx", "input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    ### some examples may exceed the max length so that their labels will be [-100...-100] but they cannot be filtered because of the index
    # invalid_indices = []
    # for idx, example in enumerate(lm_datasets['train']):
    #     if (example['labels'] == -100).all():
    #         invalid_indices.append(idx)
    # lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any()) 
    if args.with_prompt_token:
        print("*** current also use prompt token ***")
    else:
        print("*** current only use response token ***")
            
    train_dataset = lm_datasets["train"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    ) 

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True



    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=False)
    completed_steps = 0
    starting_epoch = 0

    raw_labels = train_dataset['labels']

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    model.eval()
    reference_model.eval()
    with torch.no_grad():
        for epoch in range(starting_epoch, args.num_train_epochs):
            total_loss = 0

            active_dataloader = train_dataloader
            token_logits = []
            token_kl_div_scores = []
            # for step, batch in enumerate(tqdm(active_dataloader, desc="Processing batches")):
            for step, batch in enumerate(active_dataloader):

                # with accelerator.accumulate(model):
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                indices = batch["idx"]
                outputs = model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'], use_cache=False)
                ref_outputs = reference_model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'], use_cache=False)

                if args.reduce_loss == 'mean':
                    loss = outputs.loss
                else:
                    # reduce loss is sum
                    # this ensures that we weight all tokens in the dataset equally,
                    # rather than weighting each overall example equally when
                    # using high amounts of gradient accumulation.
                    # this can result in > 5 point improvements in AlpacaEval
                    # see https://github.com/huggingface/transformers/issues/24725 for
                    # more discussion and details.
                    
                    logits = outputs.logits
                    ref_logits = ref_outputs.logits
                    # labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].half().cpu()
                    shift_ref_logits = ref_logits[..., :-1, :].half().cpu()

                    # shift_labels = labels[..., 1:].contiguous()
                    # token_shift_logits = shift_logits.half()

                    # Flatten the tokens
                    # loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                    # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

                    # shift_logits = shift_logits.view(-1, embedding_size)
                    # shift_labels = shift_labels.view(-1)
                    # # Enable model parallelism
                    # shift_labels = shift_labels.to(shift_logits.device)
                    
                    # # loss = loss_fct(shift_logits, shift_labels)
                    # per_token_loss = loss_fct(shift_logits, shift_labels)
                    # loss = per_token_loss.sum()

                    # Save per-token loss (reshape back to match sequence dimensions if needed)
                    # per_token_loss = per_token_loss.view(batch["labels"].size(0), -1)  # Shape: [batch_size, seq_len]
                    # padded_token_losses = F.pad(per_token_loss, (0, pad_length - per_token_loss.shape[-1]), value=0)
                    # padded_token_logits = F.pad(shift_logits, (0, 0, 0, pad_length - batch["labels"].shape[-1]+1), value=0)
                    
                    # token_losses.append(padded_token_losses)  # Save loss to analyze later
                    # batch_indices.append(indices)
                    kl_div_scores = compute_kl_div_score(shift_ref_logits, shift_logits, reduction='none')
                    # for i, sample_idx in enumerate(indices):
                    #     revised_logits = torch.cat([torch.zeros((1, shift_logits.shape[-1]), dtype=shift_logits.dtype, device=shift_logits.device), shift_logits[i][:len(raw_labels[sample_idx])-1]], dim=0)
                    #     token_logits.append(revised_logits)  ### [0] correct the biased weight and set the first token with loss 0
                        
                    for i, sample_idx in enumerate(indices):
                        sample_kl_div_scores = torch.cat([torch.zeros((1), dtype=kl_div_scores.dtype), kl_div_scores[i][:len(raw_labels[sample_idx])-1]], dim=0)
                        token_kl_div_scores.append(sample_kl_div_scores)  ### [0] correct the biased weight and set the first token with loss 0
                        
                    
                    progress_bar.update(1) ## tdqm
                    completed_steps += 1
                    
                    
    ## save the loss
    # logits_path = "/mnt/data1/jinlong/token_selection_output/logits/"
    kl_div_scores_path = "results/kl_div_scores/"

    if not os.path.exists(kl_div_scores_path):
        os.makedirs(kl_div_scores_path)
    
    model_name = os.path.basename(args.model_name_or_path)
    data_type= os.path.basename(args.train_file).split(".json")[0]
    final_data_path = kl_div_scores_path + f"token_kl_div_scores_{data_type}_{model_name}.pt"
    # import pdb; pdb.set_trace()
    torch.save(token_kl_div_scores, final_data_path)
    print(f"*** Token-level loss has been stored in {final_data_path} ***")

if __name__ == "__main__":
    main()