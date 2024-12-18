from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
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
    # labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
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
            # labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def get_global_top_k_indices(data, k):

    flattened = [(value, i, j) for i, sublist in enumerate(data) for j, value in enumerate(sublist)]
    
    top_k = sorted(flattened, key=lambda x: x[0], reverse=True)[:k] ##loss
    
    top_k_indices = [(item[1], item[2]+1) for item in top_k]  #item[2]+1 fix the first label biased to match the position
    return top_k_indices


def main(
    base_model=None,
    data_type=None,
    model_type='test',
    new_model_type='test',
    data_prop: float = 1.0,
    global_level_top_k_indices = False,
    sample_level_top_k_indices = False,
    union_level_top_k_indices = False,
    additional_two_tokens_level_top_k_indices = False,
    reverse_loss  = False,
    ):

    train_data=f"selected_data/{data_type}.json"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    raw_dataset = load_dataset("json", data_files=train_data)

    if "prompt" in raw_dataset["train"].column_names and "completion" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            add_bos= False,
        )
    elif "messages" in raw_dataset["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
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
    losses_pre = torch.load(f"results/loss/token_losses_{data_type}_{model_type}.pt")
    losses_cur = torch.load(f"results/loss/token_losses_{data_type}_{new_model_type}.pt")
    
    # initialize: ignore all tokens
    selected_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
    
    ##the calculation different loss of two models
    if reverse_loss:
        loss_diff = [(np.array(loss2) - np.array(loss1)).tolist() for loss1, loss2 in zip(losses_pre, losses_cur)]
    else:
        loss_diff = [(np.array(loss1) - np.array(loss2)).tolist() for loss1, loss2 in zip(losses_pre, losses_cur)]


    all_token_count = sum(len(label) for label in raw_labels)
    print(f"#### all token counting: {all_token_count}\n")

    print(f"model pair: ({model_type}, {new_model_type}) -- dataset: {data_type}")
    
    # global-level top-k data selection
    if global_level_top_k_indices: #global-level top-k
        print("### start global level top-k selection...")
        select_tokens_indices = get_global_top_k_indices(loss_diff, int(all_token_count * data_prop))
        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
        
    #sample-level top-k
    elif sample_level_top_k_indices: #sample-level top-k
        print("### start sample level top-k selection...")

        select_tokens_indices = []

        for diff in loss_diff:
            _, indices = torch.topk(torch.tensor(diff), k=int(len(diff) * data_prop), largest=True)
            select_tokens_indices.append((indices + 1).tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset
        
        for i, (selected_indices, label) in enumerate(zip(select_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_labels[i][j] = label[j]
                
    elif union_level_top_k_indices:
        print("### start union level top-k selection...")

        ### global-level
        selected_global_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
        select_global_tokens_indices = get_global_top_k_indices(loss_diff, int(all_token_count * data_prop))
        for i, j in select_global_tokens_indices:
                selected_global_labels[i][j] = raw_labels[i][j] 
    
        ### sample-level 
        selected_sample_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
        select_sample_tokens_indices = []
        for diff in loss_diff:
            _, indices = torch.topk(torch.tensor(diff), k=int(len(diff) * data_prop), largest=True)
            select_sample_tokens_indices.append((indices + 1).tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset    
        for i, (selected_indices, label) in enumerate(zip(select_sample_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_sample_labels[i][j] = label[j]
    
        ## calculate the union label
        for i, (selected_global_labels_per_sample, selected_sample_labels_per_sample) in enumerate(zip(selected_global_labels, selected_sample_labels)):
            for j, (global_label, sample_label) in enumerate(zip(selected_global_labels_per_sample, selected_sample_labels_per_sample)):
                if global_label != -100 or sample_label != -100:
                    chosen_label = global_label if global_label != -100 else sample_label
                    
                    selected_labels[i][j] = chosen_label
                    
    elif additional_two_tokens_level_top_k_indices:
        print("### start additional_two_tokens level top-k selection...")

        ### global-level
        selected_global_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
        select_global_tokens_indices = get_global_top_k_indices(loss_diff, int(all_token_count * data_prop))
        for i, j in select_global_tokens_indices:
                selected_global_labels[i][j] = raw_labels[i][j] 
                ## two more tokens
                if j + 1 < len(raw_labels[i]):
                    selected_global_labels[i][j+1] = raw_labels[i][j+1] 
                if j + 2 < len(raw_labels[i]):
                    selected_global_labels[i][j+2] = raw_labels[i][j+2] 
    
    
        ### sample-level 
        selected_sample_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
        select_sample_tokens_indices = []
        for diff in loss_diff:
            _, indices = torch.topk(torch.tensor(diff), k=int(len(diff) * data_prop), largest=True)
            select_sample_tokens_indices.append((indices + 1).tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset    
        for i, (selected_indices, label) in enumerate(zip(select_sample_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_sample_labels[i][j] = label[j]
                ### two more tokens
                if j + 1 < len(label):
                    selected_sample_labels[i][j+1] = label[j+1] 
                if j + 2 < len(label):
                    selected_sample_labels[i][j+2] = label[j+2] 
    
        ## calculate the union label
        for i, (selected_global_labels_per_sample, selected_sample_labels_per_sample) in enumerate(zip(selected_global_labels, selected_sample_labels)):
            for j, (global_label, sample_label) in enumerate(zip(selected_global_labels_per_sample, selected_sample_labels_per_sample)):
                if global_label != -100 or sample_label != -100:
                    chosen_label = global_label if global_label != -100 else sample_label
                    
                    selected_labels[i][j] = chosen_label
                    
    else:
        print("Please choose the token-level selection method: (1) global-level, (2) sample-level or (3) union-level!")
        raise NotImplementedError
    
    ### extract the sample from the original dataset and store the new dataset
    torch.save(selected_labels, f"results/label/token_labels_{data_type}.pt")



if __name__ == "__main__":
    fire.Fire(main)