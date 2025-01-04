#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
NUM_GPUS=6



start_time=$(date +%s)

BATCH_SIZE_PER_GPU=2

######################################
############ base_models #############
declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="8 1 128" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
base_models["meta-llama/Llama-3.2-3B"]="$(($BATCH_SIZE_PER_GPU*$NUM_GPUS)) ${BATCH_SIZE_PER_GPU} 2048"

# Train_DATASET_LIST=('filtered-cured-50k-all-iter-sample-subset-small-new' 'filtered-cured-50k-all-iter-global-subset-small-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-non-iter-sample-subset-new' 'filtered-cured-50k-all-non-iter-global-subset-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-iter-union-subset-small-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-iter-additional_two_tokens-subset-small-new') #

# Train_DATASET_LIST=('filtered-cured-50k-all-iter-combine-loss-subset-small-new') #
cluster_root_path="/mnt/data1/jinlong/token_selection_output"
mkdir -p $cluster_root_path


Train_DATASET_LIST=('filtered-cured-50k_all_random_select' 'filtered-cured-50k_all_loss_ranking_select') #

data_prop=0.3 #0.3
main_process_port=29516

# token_select_pattern="random_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"
# model_type_tag="random_select"


for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
    echo "##### train_dataset_name: ${train_dataset_name}"

    if [[ $train_dataset_name == *"random_select"* ]]; then
        token_select_pattern="random_select" 
        model_type_tag="random_select"

    elif [[ $train_dataset_name == *"loss_ranking_select"* ]]; then

        token_select_pattern="loss_ranking_select"
        model_type_tag="loss_ranking_select"
    else
        echo "Fail to match the training dataset name!"
    fi

    for base_model in "${!base_models[@]}"
    do
        IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
        TOTAL_BATCH_SIZE=${params[0]}
        BATCH_SIZE_PER_GPU=${params[1]}
        max_seq_length=${params[2]}

        model_name_or_path=$base_model
        # model_name_or_path=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${model_type}/

        mkdir -p $cluster_root_path/models/

        ###finetune ###
        echo "######### Start finetuning...."
        bash_src/finetune.sh "$model_name_or_path" "$model_type_tag" "$train_dataset_name" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$TOTAL_BATCH_SIZE" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"
        
    done
done



end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"


# nohup bash run_all.sh > zzz_sample_iter_subset_small_new.log &
# nohup bash run_all.sh > zzz_sample_non_iter_subset_small_new.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-union.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-additional-two-tokens.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-intersection.log &

# bash run_random_select.sh > zzz_random_select_and_loss_ranking_select.log 2>&1