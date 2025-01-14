from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire
import os

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
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
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
            labels[:, message_start_idx:message_end_idx] = -100

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
    
    top_k_indices = [(item[1], item[2]) for item in top_k]  #item[2]+1 fix the first label biased to match the position
    return top_k_indices

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
    ):
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
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

        # ### new loss diff form: w1*(loss1 -loss2) + (1-w1)*loss2 --> w1*loss1 - (2w1-1)*loss2
        # beta =0.6
        # loss_diff = [(beta * np.array(loss1) - (2 *beta -1) * np.array(loss2)).tolist()  for loss1, loss2 in zip(losses_pre, losses_cur)]
    
    
    all_token_count = sum(len(label) for label in raw_labels)
    print(f"#### all token counting: {all_token_count}\n")

    print(f"model pair: ({base_model_name}, {ref_model_name}) -- dataset: {data_type}")
    
    # global-level top-k data selection
    if select_token_level == 'global': #global-level top-k
        print("### start global level top-k selection...")
        select_tokens_indices = get_global_top_k_indices(loss_diff, int(all_token_count * data_prop))

        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    # global-level positive data selection
    elif select_token_level == 'global-positive': #global-level positive loss diff token
        print("### start global-level positive loss diff selection...")

        select_tokens_indices = get_positive_indices(loss_diff)

        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    # global-level positive data selection
    elif select_token_level == 'global-half-positive': #global-level positive loss diff token
        print("### start global-level half positive loss diff selection...")

        select_tokens_indices = get_half_positive_indices(loss_diff)
        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    elif select_token_level == 'global-curve-positive': #global-level positive loss diff token
        print("### start global-level curve positive loss diff selection...")

        select_tokens_indices = get_curve_positive_indices(losses_pre, losses_cur)

        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")        
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 

    #sample-level top-k
    elif select_token_level == 'sample': #sample-level top-k
        print("### start sample level top-k selection...")

        select_tokens_indices = []

        for diff in loss_diff:
            _, indices = torch.topk(torch.tensor(diff), k=int(len(diff) * data_prop), largest=True)
            select_tokens_indices.append(indices.tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset
        
        for i, (selected_indices, label) in enumerate(zip(select_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_labels[i][j] = label[j]
                
    #sample-level positive
    elif select_token_level == 'sample-positive':  # sample-level positive selection
        print("### start sample level positive selection...")
        select_tokens_indices = []

        for diff in loss_diff:
            positive_indices = [i for i, value in enumerate(diff) if value > 0] #i+1 represents the token bias
            select_tokens_indices.append(positive_indices)

        for i, (selected_indices, label) in enumerate(zip(select_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_labels[i][j] = label[j]    
                
                
    elif select_token_level == 'union':
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
            select_sample_tokens_indices.append(indices.tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset    
        for i, (selected_indices, label) in enumerate(zip(select_sample_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_sample_labels[i][j] = label[j]
    
        ## calculate the union label
        for i, (selected_global_labels_per_sample, selected_sample_labels_per_sample) in enumerate(zip(selected_global_labels, selected_sample_labels)):
            for j, (global_label, sample_label) in enumerate(zip(selected_global_labels_per_sample, selected_sample_labels_per_sample)):
                if global_label != -100 or sample_label != -100:
                    chosen_label = global_label if global_label != -100 else sample_label
                    
                    selected_labels[i][j] = chosen_label
                    
    elif select_token_level == 'intersection':
        print("### start intersection level top-k selection...")

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
            select_sample_tokens_indices.append(indices.tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset    
        for i, (selected_indices, label) in enumerate(zip(select_sample_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_sample_labels[i][j] = label[j]
    
        ## calculate the intersection label
        for i, (selected_global_labels_per_sample, selected_sample_labels_per_sample) in enumerate(zip(selected_global_labels, selected_sample_labels)):
            for j, (global_label, sample_label) in enumerate(zip(selected_global_labels_per_sample, selected_sample_labels_per_sample)):
                if global_label != -100 and global_label == sample_label:
                    
                    selected_labels[i][j] = global_label

    elif select_token_level == "additional_two_tokens":
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
            select_sample_tokens_indices.append(indices.tolist()) ## indices +1 represents the biased value, which match the real token in the original dataset    
        for i, (selected_indices, label) in enumerate(zip(select_sample_tokens_indices, raw_labels)):
            for j in selected_indices:
                selected_sample_labels[i][j] = label[j]
                ### two more tokens
                if j + 1 < len(label):
                    selected_sample_labels[i][j+1] = label[j+1] 
                if j + 2 < len(label):
                    selected_sample_labels[i][j+2] = label[j+2] 
    
        ## calculate the add_two_tokens label
        for i, (selected_global_labels_per_sample, selected_sample_labels_per_sample) in enumerate(zip(selected_global_labels, selected_sample_labels)):
            for j, (global_label, sample_label) in enumerate(zip(selected_global_labels_per_sample, selected_sample_labels_per_sample)):
                if global_label != -100 or sample_label != -100:
                    chosen_label = global_label if global_label != -100 else sample_label
                    
                    selected_labels[i][j] = chosen_label
                    
    elif select_token_level == "combine_loss": #global-level top-k + loss_top-k
        print("### start combine_loss selection...")
        select_tokens_indices = get_global_top_k_indices(loss_diff, int(all_token_count * data_prop))
        
        ### add solely losses_cur top-k tokens to avoid tradeoff
        select_tokens_indices_cur = get_global_top_k_indices(losses_cur, int(all_token_count * data_prop))
        select_tokens_indices +=select_tokens_indices_cur
        
        select_sample_idx = [item[0] for item in select_tokens_indices]
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")  
        for i, j in select_tokens_indices:
                selected_labels[i][j] = raw_labels[i][j] 
                
    elif select_token_level == "token_ranking_sample_select": # use the top-30% token for each sample for sample-level ranking; take 30% samples with all tokens to finetune
        print("### start token_ranking_sample_select selection...")
        # select_tokens_indices = get_curve_positive_indices(losses_pre, losses_cur)
        select_tokens_indices = get_positive_indices(loss_diff)
        
        select_sample_idx = [item[0] for item in select_tokens_indices]
        
        from collections import Counter
        selected_num_tokens_per_sample = sorted(Counter(select_sample_idx).items(), key=lambda x: x[1], reverse=True)
        
        sample_prop = 0.3
        selected_num_examples = min(int(len(raw_labels) * sample_prop), len(selected_num_tokens_per_sample))
            
        selected_sample_indices = [key for key, value in selected_num_tokens_per_sample[:selected_num_examples]]
            
            
        select_sample_idx = set(select_sample_idx)
        print(f"selected sample size:: {len(select_sample_idx)} -- original dataset size: {len(raw_labels)}")    
        print(f"current selected sample size: {len(selected_sample_indices)}")
        ## select samples
        for i in selected_sample_indices:
                selected_labels[i] = raw_labels[i]
                       
    else:
        print("Please choose the token-level selection method from: (1) global, (2) sample, (3) union, (4) intersection (5) additional_two_tokens or (6) combine_loss!")
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