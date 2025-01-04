

model_name_or_path=$1
model_type=$2
data_type=$3
max_seq_length=$4
BATCH_SIZE_PER_GPU=$5
NUM_GPUS=$6
base_model=$7
TOTAL_BATCH_SIZE=$8
cluster_root_path=$9
data_prop=${10}
main_process_port=${11}
token_select_pattern=${12}

train_data="selected_data/${data_type}.json"

GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"
echo "Training data path: ${train_data}"
echo "main_process_port: ${main_process_port}"
echo "data_prop: ${data_prop}"


accelerate launch \
    --num_machines 1 \
    --mixed_precision bf16 \
    --num_processes $NUM_GPUS \
    --config_file fsdp_configs/fsdp_config.yaml \
    --main_process_port $main_process_port \
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
    --output_dir $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${data_type}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --model_type $model_type \
    --data_type $data_type \
    --token_select_pattern $token_select_pattern \
    --data_prop $data_prop
    # --reduce_loss sum \

python open_instruct/merge_lora.py \
    --base_model_name_or_path $model_name_or_path \
    --lora_model_name_or_path $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${data_type}/ \
    --output_dir $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${data_type}/ \
    --save_tokenizer

sleep 10s
rm -rf $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${data_type}