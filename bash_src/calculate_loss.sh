#!/bin/bash

model_name_or_path=$1
model_type=$2
data_type=$3
max_seq_length=$4
BATCH_SIZE_PER_GPU=$5
NUM_GPUS=$6
main_process_port=$7

train_data="selected_data/${data_type}.json"

echo "train_data: ${train_data}"
echo "main_process_port: ${main_process_port}"


accelerate launch \
    --num_processes $NUM_GPUS \
    --config_file fsdp_configs/fsdp_config.yaml \
    --main_process_port $main_process_port \
    open_instruct/calculate_token_loss.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_data \
    --max_seq_length $max_seq_length \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --num_train_epochs 1 \
    --reduce_loss sum \
    --model_type $model_type \
    --data_type $data_type 

