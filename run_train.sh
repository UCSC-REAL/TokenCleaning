#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8


BATCH_SIZE_PER_GPU=3

######################################
############ base_models #############
declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="8 1 128" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
base_models["meta-llama/Llama-3.2-3B"]="$(($BATCH_SIZE_PER_GPU*$NUM_GPUS)) ${BATCH_SIZE_PER_GPU} 2048"


## # model_types used for ablation study, which determines the finetuned model
model_types=('base')

data_type="filtered-cured"

#############################################################
######## model finetuning on selected training data ######### 
#############################################################

echo "###### All data types here:: ${model_types[@]}"
echo "###### All training datasets here:: ${TRAIN_DATASET_LIST[@]}"


# cluster_root_path="output" ## . for local
cluster_root_path="/mnt/data1/jinlong/token_selection_output"

mkdir -p $cluster_root_path

for base_model in "${!base_models[@]}"
do
    IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
    TOTAL_BATCH_SIZE=${params[0]}
    BATCH_SIZE_PER_GPU=${params[1]}
    max_seq_length=${params[2]}


    for model_type in "${model_types[@]}"
    do
        echo "###### Processing model type:: ${model_type}"

        if [[ $model_type == "base" ]]; then
            model_name_or_path=$base_model
        else
            model_name_or_path=$cluster_root_path/models/${base_model}/lora_merged_${model_type}/
        fi

        mkdir -p $cluster_root_path/models/

        train_data="selected_data/${data_type}_dataset.json"

        GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
        echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
        echo "Training data path: ${train_data}"

        accelerate launch \
            --num_machines 1 \
            --mixed_precision bf16 \
            --num_processes $NUM_GPUS \
            --config_file fsdp_configs/fsdp_config.yaml \
            --main_process_port 29503 \
            open_instruct/finetune.py \
            --model_name_or_path $model_name_or_path \
            --gradient_checkpointing \
            --use_lora \
            --lora_rank 64 \
            --lora_alpha 16 \
            --lora_dropout 0.1 \
            --tokenizer_name $model_name_or_path \
            --use_slow_tokenizer \
            --train_file $train_data \
            --max_seq_length $max_seq_length \
            --preprocessing_num_workers 16 \
            --checkpointing_steps epoch \
            --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
            --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
            --learning_rate 1e-4 \
            --lr_scheduler_type linear \
            --warmup_ratio 0.03 \
            --weight_decay 0. \
            --num_train_epochs 1 \
            --output_dir $cluster_root_path/models/${base_model}/lora_${data_type}/ \
            --with_tracking \
            --report_to tensorboard \
            --logging_steps 1 \
            --reduce_loss sum

        python open_instruct/merge_lora.py \
            --base_model_name_or_path $model_name_or_path \
            --lora_model_name_or_path $cluster_root_path/models/${base_model}/lora_${data_type}/ \
            --output_dir $cluster_root_path/models/${base_model}/lora_merged_${model_type}/ \
            --save_tokenizer

        sleep 10s

        rm -rf $cluster_root_path/models/${base_model}/lora_${data_type}/

    done
done

echo "finish model traning!!"