from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire
import os
import torch.nn.functional as F

def compute_kl_div_score(logits_pre, logits_cur, reduction='batchmean'):
    """
    计算两个模型在整个序列上的 KL 散度 (Kullback-Leibler Divergence)。

    参数:
        logits_pre (torch.Tensor): 之前模型的 logits, 形状 (batch_size, seq_len, vocab_size)。
        logits_cur (torch.Tensor): 当前模型的 logits, 形状 (batch_size, seq_len, vocab_size)。
        reduction (str): KL 散度的 reduction 方法 ('batchmean', 'sum', 'mean', 'none')。

    返回:
        torch.Tensor: KL 散度分数，形状 (batch_size, seq_len) 或标量。
    """
    # 确保两个 logits 形状一致
    assert logits_pre.shape == logits_cur.shape, "logits_pre 和 logits_cur 形状必须相同"

    # 计算 log-probabilities (log softmax)
    log_probs_pre = F.log_softmax(logits_pre, dim=-1)  # 之前模型的 log 概率
    probs_cur = F.softmax(logits_cur, dim=-1)  # 当前模型的普通概率

    # 计算 KL 散度（每个 token 计算一次）
    kl_div = F.kl_div(log_probs_pre, probs_cur, reduction=reduction)

    return kl_div

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



def get_global_top_k_indices(raw_labels, all_losses, data_prop):

    response_tokens = []
    for i, (sample_labels, sample_losses) in enumerate(zip(raw_labels, all_losses)):
        for j, (label, loss) in enumerate(zip(sample_labels, sample_losses)):
            if label !=-100:
                response_tokens.append((loss, i, j))
    
    top_k_tokens = sorted(response_tokens, key=lambda x: x[0], reverse=True)[:int(len(response_tokens)*data_prop)] ##loss
    
    top_k_indices = [(item[1], item[2]) for item in top_k_tokens]  
    return top_k_indices


def get_sample_top_k_indices(raw_labels, all_losses, data_prop):

    response_tokens_indices = []
    for i, (sample_labels, sample_losses) in enumerate(zip(raw_labels, all_losses)):
        response_tokens_per_sample = []
        for j, (label, loss) in enumerate(zip(sample_labels, sample_losses)):
            if label !=-100:
                response_tokens_per_sample.append((loss, i, j))
                
        top_k_tokens_per_sample = sorted(response_tokens_per_sample, key=lambda x: x[0], reverse=True)[:int(len(response_tokens_per_sample)*data_prop)] ##loss
    
        top_k_indices_per_sample = [(item[1], item[2]) for item in top_k_tokens_per_sample] 
        response_tokens_indices.extend(top_k_indices_per_sample)
        
    return response_tokens_indices


def get_positive_indices(data):
    positive_indices = [(i, j) for i, sublist in enumerate(data) for j, value in enumerate(sublist) if value > 0]
    return positive_indices

def get_half_positive_indices(data):
    selected_flattened = [(value, i, j) for i, sublist in enumerate(data) for j, value in enumerate(sublist) if value > 0]
    top_half_positive = sorted(selected_flattened, key=lambda x: x[0], reverse=True)[:int(len(selected_flattened)/2)] ##loss

    top_half_positive_indices = [(item[1], item[2]) for item in top_half_positive]
    return top_half_positive_indices

def  get_curve_positive_indices(losses_pre, losses_cur, alpha = 2, beta = 0.07, threshold=5):
    #alpha, beta = 1.2, 0.1
    #alpha, beta = 1.5, 0.5    
    curve_positive_indices=[]
    
    for i, (sample_losses_pre, sample_losses_cur) in enumerate(zip(losses_pre, losses_cur)):
        for j, (token_loss_pre, token_loss_cur) in enumerate(zip(sample_losses_pre, sample_losses_cur)):
            if token_loss_pre > alpha * token_loss_cur + beta and token_loss_cur < threshold: #linear split
                curve_positive_indices.append((i, j))

    return curve_positive_indices


def  get_curve_smooth_positive_indices(losses_pre, losses_cur, subset_idx, num_subset, alpha_base = 2, beta_base = 0.07, threshold=5):
    #alpha, beta = 1.2, 0.1
    #alpha, beta = 1.5, 0.5   
    #alpha, beta = 2, 0.07  

    alpha = alpha_base - subset_idx / (num_subset-1) * 0.9
    beta = beta_base - subset_idx / (num_subset-1)  * 0.05
    
    curve_positive_indices=[]
    
    for i, (sample_losses_pre, sample_losses_cur) in enumerate(zip(losses_pre, losses_cur)):
        for j, (token_loss_pre, token_loss_cur) in enumerate(zip(sample_losses_pre, sample_losses_cur)):
            if token_loss_pre > alpha * token_loss_cur + beta and token_loss_cur < threshold: #linear split
                curve_positive_indices.append((i, j))

    return curve_positive_indices


def get_fixed_positive_indices(data, k):
    selected_flattened = [(value, i, j) for i, sublist in enumerate(data) for j, value in enumerate(sublist) if value > 0]
    top_half_positive = sorted(selected_flattened, key=lambda x: x[0], reverse=True)[:min(k, len(selected_flattened))] ##loss

    top_half_positive_indices = [(item[1], item[2]) for item in top_half_positive]
    return top_half_positive_indices

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise NotImplementedError


def main(
    base_model_name_or_path='test',
    ref_model_name_or_path='test',
    train_data=None,
    data_prop: float = 1.0,
    select_token_level="sample",
    subset_idx = 0,
    num_subset = 5,
    label_path = "results/label/",
    loss_path = "results/loss/",
    reverse_loss = False,
    with_prompt_token=False,
    valid_dataset_name='mmlu',
    ):
    
    if with_prompt_token:
        print("current also use prompt token")
    else:
        print("current use only response token")
        
    # import pdb;pdb.set_trace()
    if "lora" not in base_model_name_or_path or os.path.exists(base_model_name_or_path): ## means huggingface model or existed local model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    else:
        if "mistral" in base_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        elif "llama3b" in base_model_name_or_path or "llama8b" in base_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        else:
            print("unknown model.")
    
    raw_dataset = load_dataset("json", data_files=train_data)

    ### rename
    base_model_name = os.path.basename(base_model_name_or_path)
    ref_model_name = os.path.basename(ref_model_name_or_path)
    data_type= os.path.basename(train_data).split(".json")[0]


    if "prompt" in raw_dataset["train"].column_names and "completion" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            with_prompt_token = with_prompt_token,
            add_bos= False,
        )
    elif "messages" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            with_prompt_token = with_prompt_token,
            add_bos= False,
        )
        
    raw_dataset = raw_dataset.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,  
        desc="Adding idx column",
    )
            

    lm_datasets = raw_dataset.map(
        encode_function,
        batched=False,
        # remove_columns=[name for name in raw_dataset["train"].column_names if name not in ["idx", "input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )

    train_dataset = lm_datasets['train']
    raw_labels = train_dataset['labels']
    if with_prompt_token:
        print("*** current also use prompt token ***")
    

    selected_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
    
    #####calculate the kl-divergence score between two logits
    # logits_pre = torch.load(logits_path_pre)
    # logits_cur = torch.load(logits_path_cur)
    
    # kl_divergence_scores = compute_kl_div_score(logits_pre, logits_cur)
    
    # infl_scores = torch.load(f"final_token_infl_scores/ds2_token_infl_scores_{valid_dataset_name}.pt")
    kl_divergence_scores = torch.load(f"results/kl_div_scores/token_kl_div_scores_filtered-cured-50k-rho-baseline-global-llama3b-kl-divergence_Llama-3.2-3B.pt")
    # all_token_count = sum(len(label) for label in raw_labels)
    all_token_count = sum(1 for labels_per_sample in raw_labels for label in labels_per_sample if label != -100)
    
    print(f"#### all token counting (prompt + response): {sum(len(label) for label in raw_labels)}\n")
    print(f"#### all token counting (response): {all_token_count}\n")

    print(f"model pair: ({base_model_name}, {ref_model_name}) -- dataset: {data_type}")
    
    # global-level top-k data selection
    if select_token_level == 'global': #global-level top-k
        print("### start global level top-k selection...")
        select_tokens_indices = get_global_top_k_indices(raw_labels, kl_divergence_scores, data_prop)

        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
    #sample-level top-k
    elif select_token_level == 'sample': #sample-level top-k
        print("### start sample level top-k selection...")
        select_tokens_indices=get_sample_top_k_indices(raw_labels, kl_divergence_scores, data_prop)
        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
           
    else:
        print("Please choose the token-level selection method from: (1) global, (2) sample, (3) union, (4) intersection (5) additional_two_tokens or (6) combine_loss!")
        raise NotImplementedError
    
    
    
    ## save the loss
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    ### extract the sample from the original dataset and store the new dataset
    final_data_path = label_path + f"token_labels_{data_type}.pt"
    torch.save(selected_labels, final_data_path)
    import pdb;pdb.set_trace()
    print(f"*** Token-level label has been stored in {final_data_path} ***")


if __name__ == "__main__":
    fire.Fire(main)