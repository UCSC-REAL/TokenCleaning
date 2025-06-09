from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire
import os

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
    ):
        
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
    
    ################ current train model loss ##############
    # if "Llama-3.2-3B" in base_model_name: ## load from existing model
    #     if "filtered-cured-50k" in data_type:
    #         base_loss_path = loss_path + f"token_losses_filtered-cured-50k_all_{base_model_name}.pt" 
    #     elif "random_subset_50k" in data_type:
    #         base_loss_path = loss_path + f"token_losses_random_subset_50k_all_{base_model_name}.pt"    
    #     else:
    #         print("unknow dataset, please check whether generate the loss for base model.")
    #         raise NotImplementedError
        
    #     print(f"load the first round base model from existing file: {base_loss_path}")
    #     losses_pre = torch.load(base_loss_path)[:len(raw_labels)]

    # else:
    #     losses_pre = torch.load(loss_path + f"token_losses_{data_type}_{base_model_name}.pt")
    
    
    # ############### reference model loss #############
    # if "filtered-cured-50k" in data_type and ref_model_name == "Llama-3.1-8B-Instruct":
    #     reference_loss_path = loss_path + f"token_losses_filtered-cured-50k_all_{ref_model_name}.pt"
    # elif "random_subset_50k" in data_type and ref_model_name == "Llama-3.1-8B-Instruct":
    #     reference_loss_path = loss_path + f"token_losses_random_subset_50k_all_{ref_model_name}.pt"
    # else:
    #     reference_loss_path = None
        
    # ### reuse the existing reference loss
    # if  reference_loss_path and os.path.exists(reference_loss_path):
    #     print(f"load the reference losses from existing file: {reference_loss_path}")
    #     all_losses = torch.load(reference_loss_path)
    #     subset_size = int(len(all_losses) / num_subset)
    #     losses_cur = all_losses[subset_idx*subset_size:(subset_idx+1)*subset_size]
    # else:
    #     losses_cur = torch.load(f"results/loss/token_losses_{data_type}_{ref_model_name}.pt")
        
        
    ### original loss ####
    losses_pre = torch.load(loss_path + f"token_losses_{data_type}_{base_model_name}.pt")
    losses_cur = torch.load(loss_path + f"token_losses_{data_type}_{ref_model_name}.pt")

    # initialize: ignore all tokens first
    selected_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
    ##the calculation different loss of two models
    if reverse_loss:
        loss_diff = [(np.array(loss2) - np.array(loss1)).tolist() for loss1, loss2 in zip(losses_pre, losses_cur)]
    else:
        loss_diff = [(np.array(loss1) - np.array(loss2)).tolist() for loss1, loss2 in zip(losses_pre, losses_cur)]

    
    # all_token_count = sum(len(label) for label in raw_labels)
    all_token_count = sum(1 for labels_per_sample in raw_labels for label in labels_per_sample if label != -100)
    
    print(f"#### All token counting (prompt + response): {sum(len(label) for label in raw_labels)}\n")
    print(f"#### All token counting (response): {all_token_count}\n")

    print(f"Current model pair: ({base_model_name}, {ref_model_name}) -- dataset: {data_type}")
    
    # global-level top-k data selection
    if select_token_level == 'global': 
        print("### Global level top-k selection...")
        select_tokens_indices = get_global_top_k_indices(raw_labels, loss_diff, data_prop)

        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    #sample-level top-k
    elif select_token_level == 'sample':
        print("### Sample level top-k selection...")
        select_tokens_indices=get_sample_top_k_indices(raw_labels, loss_diff, data_prop)
        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    else:
        print("Please choose the token-level selection method from: (1) global or (2) sample")
        raise NotImplementedError
    
    
    ## save the loss
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    ### extract the sample from the original dataset and store the new dataset
    final_data_path = label_path + f"token_labels_{data_type}.pt"
    torch.save(selected_labels, final_data_path)

    print(f"*** Token-level label has been stored in {final_data_path} ***")


if __name__ == "__main__":
    fire.Fire(main)