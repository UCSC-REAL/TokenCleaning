#!/bin/bash

model_name_or_path=$1
train_data=$2
BATCH_SIZE_PER_GPU=$3
NUM_GPUS=$4

accelerate launch \
    --num_processes $NUM_GPUS \
    --config_file fsdp_configs/fsdp_config.yaml \
    --main_process_port 29501 \
    scripts/calculate_token_loss.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_data \
    --max_seq_length 2048 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --num_train_epochs 1 \
    --reduce_loss sum \
    --with_prompt_token False
