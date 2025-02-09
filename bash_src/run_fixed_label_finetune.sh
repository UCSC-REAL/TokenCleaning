# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8



#### basic config
max_seq_length=2048
main_process_port=29527
cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model=$1
token_select_pattern=$2
train_dataset_name=$3
data_prop=$4
BATCH_SIZE_PER_GPU=$5
reference_model=$6

echo "*** current train dataset name: ${train_dataset_name} ***"

train_data_tag_list=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")

echo "*** current data prop value: ${data_prop} ***"

for idx in "${!train_data_tag_list[@]}"; do

    train_data_tag=${train_data_tag_list[${idx}]}
    train_data="selected_data/${train_data_tag}.json"

    if [[ $idx -eq 0 ]]; then
        # cur_train_model=$base_model
        # echo "start warm-up round finetuning..."
        # warmup_token_select_pattern="all_token_select"
        # bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$warmup_token_select_pattern"
        echo "skip warmup model finetuning"
    else
        if [[ $idx -eq 1 ]]; then
            cur_train_model=$reference_model
        else
            cur_train_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-1))]}
        fi
        # Run finetune.sh script
        echo "start finetuning..."
        bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"

    fi
done


