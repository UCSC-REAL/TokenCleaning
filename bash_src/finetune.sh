
model_name_or_path=$1
train_data=$2
max_seq_length=$3
BATCH_SIZE_PER_GPU=$4
NUM_GPUS=$5
base_model=$6
cluster_root_path=$7
data_prop=$8
main_process_port=$9
token_select_pattern=${10}
with_prompt_token=${11:-False}
random_seed=${12:-42}

train_data_tag=$(basename "$train_data" .json)

# TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))
# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

GRADIENT_ACC_STEPS=1
echo "*** Training ${base_model} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps ***"
echo "*** Training data path: ${train_data} ***"
echo "*** Main_process_port: ${main_process_port} ***"
echo "*** Selected Data Proportion: ${data_prop} ***"
echo "*** With Prompt Token: ${with_prompt_token} ***"
echo "*** Random Seed: ${random_seed} ***"


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
    --output_dir $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${train_data_tag}/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --train_data_tag $train_data_tag \
    --token_select_pattern $token_select_pattern \
    --data_prop $data_prop \
    --with_prompt_token $with_prompt_token \
    --seed $random_seed 
    # --use_slow_tokenizer 

python open_instruct/merge_lora.py \
    --base_model_name_or_path $model_name_or_path \
    --lora_model_name_or_path $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${train_data_tag}/ \
    --output_dir $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag}/ \
    --save_tokenizer \
    --use_fast_tokenizer 

# sleep 10s
# rm -rf $cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_${train_data_tag}