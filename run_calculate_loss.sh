#!/bin/bash

export CUDA_VISIBLE_DEVICES=6,7
NUM_GPUS=2

######################################
############ base_models #############

declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="8 1 128" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
# base_models["mistralai/Mistral-7B-v0.3"]="128 4 2048"
base_models["meta-llama/Llama-3.2-3B"]="1 1 2048"

## # model_types used for ablation study, which determines the finetuned model
model_types=('base')
data_type=test


cluster_root_path="output" ## . for local
mkdir -p $cluster_root_path


for base_model in "${!base_models[@]}"
do
    IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
    TOTAL_BATCH_SIZE=${params[0]}
    BATCH_SIZE_PER_GPU=${params[1]}
    max_seq_length=${params[2]}

    for model_type in "${model_types[@]}"
    do

        if [[ $model_type == "base" ]]; then
            model_name_or_path=$base_model
        else
            model_name_or_path=$cluster_root_path/models/${base_model}/lora_merged_${model_type}/
        fi

        mkdir -p $cluster_root_path/models/
        train_data="selected_data/${data_type}_dataset.json"

        accelerate launch \
            --num_processes $NUM_GPUS \
            --config_file fsdp_configs/fsdp_config.yaml \
            --main_process_port 29503 \
            open_instruct/calculate_token_loss.py \
            --model_name_or_path $model_name_or_path \
            --tokenizer_name $model_name_or_path \
            --train_file $train_data \
            --max_seq_length $max_seq_length \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --num_train_epochs 1 \
            --output_dir $cluster_root_path/models/${base_model}/lora_${model_type}/ \
            --reduce_loss sum \
            --model_type $model_type \
            --data_type $data_type
    done
done

