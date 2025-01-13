# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


start_time=$(date +%s)

#### basic config
max_seq_length=2048
BATCH_SIZE_PER_GPU=3 #3
main_process_port=29520
cluster_root_path="/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"

# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/lora_merged_reference_model"
reference_model="meta-llama/Llama-3.1-8B-Instruct"

select_token_level=token_ranking_sample_select ## token_ranking_sample_select global global-positive sample-positive sample union intersection  additional_two_tokens  combine_loss
token_select_pattern="semi_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

### training data
# train_dataset_name="filtered-cured-50k-active-split"
# train_dataset_name="random_subset_50k-active-split"
# train_dataset_name="alpaca_52k-active-split"
# train_dataset_name="alpaca_52k-active-split-5k"
# train_dataset_name="full-300k-active-split-sample"
# train_dataset_name="filtered-cured-50k-active-split-global"

# train_dataset_name="filtered-cured-10k-active-split-global"

# train_dataset_name="filtered-cured-50k-active-split-global-positive"
# train_dataset_name="filtered-cured-50k-active-split-sample-positive"

# train_dataset_name="filtered-cured-50k-active-split-global-half-positive"

# train_dataset_name="filtered-cured-10k-active-split-global-curve-positive"

# train_dataset_name="filtered-cured-50k-active-split-global-curve-positive-new"
# train_dataset_name="random_subset_50k-active-split-global-curve-positive-new"



sleep 4h

train_dataset_name="filtered-cured-50k-active-split-token_ranking_sample"

# train_dataset_name="test_100-active-split-token_ranking_sample"

train_data_tag_list=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")

#### reference baseline: RHO one-shot
# train_dataset_name="random_subset_50k_active_all"
# train_data_tag_list=($train_dataset_name)

# data_prop_list=(0.3 0.6)
data_prop_list=(0.6) # if means nothing when we use the global-positive or sample-positive


for data_prop in ${data_prop_list[@]}; do
    echo "*** current data prop value: ${data_prop} ***"

    for idx in "${!train_data_tag_list[@]}"; do

        train_data_tag=${train_data_tag_list[${idx}]}
        train_data="selected_data/${train_data_tag}.json"

        if [[ $idx -eq 0 ]]; then
            cur_train_model=$base_model
        else
            cur_train_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-1))]}
        fi

        #### Run calculate_loss.sh script for base model
        # if [[ $idx -ne 0 ]]; then
        #     echo "start calculating loss for model: ${cur_train_model}"
        #     BATCH_SIZE_PER_GPU=3
        #     bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"        # # Run calculate_loss.sh script for reference model
        # else
        #     echo "skip first-round base model loss calculation. load base model loss from existing file. Current support data: filtered-cured-50k and random_subset_50k"
        # fi

        echo "start calculating loss for model: ${cur_train_model}"
        BATCH_SIZE_PER_GPU=3
        bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"        # # Run calculate_loss.sh script for reference model
        
        # ignore it when we have all reference token loss
        echo "start calculating loss for reference model: ${reference_model}"
        BATCH_SIZE_PER_GPU=2
        bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

        ## Run Python script to generate data
        echo "start generating labels.."
        python open_instruct/generate_token_label.py \
            --base_model_name_or_path $cur_train_model \
            --ref_model_name_or_path $reference_model \
            --train_data $train_data \
            --data_prop $data_prop \
            --select_token_level $select_token_level \
            --subset_idx $idx \
            --num_subset ${#train_data_tag_list[@]}

        # # Define paths for finetuning
        BATCH_SIZE_PER_GPU=3
        # Run finetune.sh script
        echo "start finetuning..."
        bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"

    done 
done


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"


# bash run_active_ref_model.sh > zzz_llama_3_8b_active_1k_samples.log 2>&1
# bash run_active_ref_model.sh > zzz_llama_3_8b_active_10k_samples.log 2>&1

# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_active_10k_samples.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_random_subset_active_10k_samples.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_alpaca_52k_active_split.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_alpaca_52k_active_split-5k.log &

# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_filtered-cured-50k-active-split-global-positive.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_filtered-cured-50k-active-split-global-half-positive.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_filtered-cured-50k-active-split-global-curve-positive-new.log &
# nohup bash run_active_ref_model.sh > zzz_llama_3_8b_random_subset_50k-active-split-global-curve-positive-new.log &
# nohup bash run_active_ref_model_token_ranking_sample.sh > zzz_llama_3_8b_random_subset_50k-active-split-token_ranking.log &