#!/bin/bash

model_name_or_path=$1
ref_model_name_or_path=$2
train_data=$3
max_seq_length=$4
BATCH_SIZE_PER_GPU=$5
NUM_GPUS=$6
main_process_port=$7
with_prompt_token=${8:-False}

echo "train_data: ${train_data}"
echo "main_process_port: ${main_process_port}"
echo "with prompt token: ${with_prompt_token}"

CUDA_VISIBLE_DIVICES=0 python open_instruct/calculate_token_logits.py \
    --model_name_or_path $model_name_or_path \
    --ref_model_name_or_path $ref_model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_data \
    --max_seq_length $max_seq_length \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --num_train_epochs 1 \
    --reduce_loss sum \
    --with_prompt_token $with_prompt_token
