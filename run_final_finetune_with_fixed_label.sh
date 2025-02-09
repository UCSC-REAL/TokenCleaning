# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


start_time=$(date +%s)

#### basic config
max_seq_length=2048
BATCH_SIZE_PER_GPU=3 #3
main_process_port=29527
cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/lora_merged_reference_model"
# reference_model="meta-llama/Llama-3.1-8B-Instruct"

select_token_level=global ## global global-positive sample-positive sample union intersection  additional_two_tokens  combine_loss

token_select_pattern=semi_select #"semi_combine_global_half_positive_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

### training data

# combine global half positive fixed base loss

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_combine_global_half_positive_fixed_based_loss")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_combine_active-split-global-curve-positive-fixed-base-loss-using-warmup-label" "filtered-cured-50k-iter-split-global_data_prop_combine_active-split-global-curve-positive-fixed-base-loss-label")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label" "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-label")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_other_two_types_label" "filtered-cured-50k-iter-split-global_data_prop_0.6_combine_other_two_types_label")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral")

Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-mistral")


data_prop_list=(0.3) # if means nothing when we use the positive series

for train_dataset_name in ${Train_DATASET_LIST[@]}; do

    echo "*** current train dataset name: ${train_dataset_name} ***"

    if [[ "$train_dataset_name" == *"0.3"* ]]; then
        data_prop_list=(0.3)
    elif [[ "$train_dataset_name" == *"0.6"* ]]; then
        data_prop_list=(0.6)
    fi

    echo "*** data_prop_list: ${data_prop_list} ***"

    train_data_tag_list=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")

    for data_prop in ${data_prop_list[@]}; do
    
        echo "*** current data prop value: ${data_prop} ***"

        for idx in "${!train_data_tag_list[@]}"; do
        # for (( idx=1; idx<${#train_data_tag_list[@]}; idx++ )); do

            train_data_tag=${train_data_tag_list[${idx}]}
            train_data="selected_data/${train_data_tag}.json"

            if [[ $idx -eq 0 ]]; then
                cur_train_model=$base_model

                # # Define paths for finetuning
                BATCH_SIZE_PER_GPU=6
                # Run finetune.sh script
                echo "start warm-up round finetuning..."
                warmup_token_select_pattern="all_token_select"
                bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$warmup_token_select_pattern"

            else
                cur_train_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-1))]}

                if [[ $idx -eq 1 ]]; then
                    reference_model=$base_model
                else
                    reference_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-2))]}
                fi

                # Run finetune.sh script
                echo "start finetuning..."
                BATCH_SIZE_PER_GPU=6
                bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"

            fi
        done 
    done

done


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $minutes minutes"


# bash run_final_finetune_with_fixed_label.sh > zzz_mistral_fixed_label.log 2>&1
# bash run_final_finetune_with_fixed_label.sh > zzz_llama8b_fixed_label.log 2>&1

# bash run_final_finetune_with_fixed_label.sh > zzz_llama8b_warmup_fixed_label.log 2>&1
# bash run_final_finetune_with_fixed_label.sh > zzz_mistral_warmup_fixed_label.log 2>&1

