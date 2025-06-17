# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


start_time=$(date +%s)

#### basic config
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29527
cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

with_prompt_token=False
select_token_level=global # sample
token_select_pattern=token_cleaning #random


### training data
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered-with-prompt" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-with-prompt")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model")

Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_mistral-non-filtered-fixed-base-model")


# Train_DATASET_LIST=("full-300k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model" )

# data_prop_list=(0.3) # if means nothing when we use the positive series


for train_dataset_name in ${Train_DATASET_LIST[@]}; do

    echo "*** current train dataset name: ${train_dataset_name} ***"

    if [[ "$train_dataset_name" == *"mistral"* ]]; then
        base_model_tag=mistral
        base_model="mistralai/Mistral-7B-v0.3"
        BATCH_SIZE_PER_GPU_finetune=6
    elif [[ "$train_dataset_name" == *"llama3b"* ]]; then
        base_model_tag=llama3b
        base_model="meta-llama/Llama-3.2-3B"
        BATCH_SIZE_PER_GPU_finetune=6
    elif [[ "$train_dataset_name" == *"llama8b"* ]]; then
        base_model_tag=llama8b
        BATCH_SIZE_PER_GPU_finetune=5
        base_model="meta-llama/Llama-3.1-8B"
    fi

    echo "*** base model: ${base_model} ***"
    echo "*** base model tag: ${base_model_tag} ***"


    reference_model=$base_model

    echo "*** subset json file generation ***"
    python scripts/generate_subset.py --generate_train_data_name $train_dataset_name

    if [[ "$train_dataset_name" == *"0.3"* ]]; then
        data_prop_list=(0.3)
    elif [[ "$train_dataset_name" == *"0.6"* ]]; then
        data_prop_list=(0.6)
    elif [[ "$train_dataset_name" == *"0.7"* ]]; then
        data_prop_list=(0.7)
    elif [[ "$train_dataset_name" == *"0.8"* ]]; then
        data_prop_list=(0.8)
    else
        echo "unknown data prop list"
    fi

    if [[ "$train_dataset_name" == *"prompt"* ]]; then
        with_prompt_token=True
    else
        with_prompt_token=False
    fi

    echo "*** data_prop_list: ${data_prop_list} ***"

    train_data_tag_list=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")

    for data_prop in ${data_prop_list[@]}; do
        echo "*** current data prop value: ${data_prop} ***"

        for idx in "${!train_data_tag_list[@]}"; do

            train_data_tag=${train_data_tag_list[${idx}]}
            train_data="selected_data/${train_data_tag}.json"

            if [[ $idx -eq 0 ]]; then
                cur_train_model=$base_model

                # # Define paths for finetuning
                warmup_token_select_pattern="all_token_select"
                BATCH_SIZE_PER_GPU=6
                # Run finetune.sh script
                echo "skip warm-up round finetuning..."
                bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$warmup_token_select_pattern" "$with_prompt_token"

            else
                cur_train_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-1))]}
                if [[ $idx -eq 1 ]]; then
                    reference_model=$base_model
                else
                    reference_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_data_tag_list[$((idx-2))]}
                fi

                #### Run calculate_loss.sh script for base model
                echo "start calculating loss for model: ${cur_train_model}"
                BATCH_SIZE_PER_GPU=6
                bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"      # # Run calculate_loss.sh script for reference model

                # ## ignore it when we have all reference token loss
                echo "start calculating loss for reference model: ${reference_model}"
                BATCH_SIZE_PER_GPU=6
                bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"

                ## Run Python script to generate data
                ### use reverse loss take the previous model as the reference model
                echo "start generating labels.."
                python scripts/generate_token_label.py \
                    --base_model_name_or_path $cur_train_model  \
                    --ref_model_name_or_path $reference_model \
                    --train_data $train_data \
                    --data_prop $data_prop \
                    --select_token_level $select_token_level \
                    --subset_idx $idx \
                    --num_subset ${#train_data_tag_list[@]} \
                    --reverse_loss True \
                    --with_prompt_token $with_prompt_token

                # Run finetune.sh script
                echo "start finetuning..."
                BATCH_SIZE_PER_GPU=$BATCH_SIZE_PER_GPU_finetune
                bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"

            fi
        done 
    done

done


end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $minutes minutes"


# bash run_iter_pattern.sh > zzz_filtered-cured-50k-iter-split-global_data_prop.log 2>&1
# bash run_iter_pattern.sh > zzz_filtered-cured-50k-iter-split-global_data_prop_combine_global_half_positive_fixed_based_loss.log 2>&1
# bash run_iter_pattern.sh > zzz_filtered-cured-50k-iter-split-global_data_prop_0.3.log 2>&1
# bash run_iter_pattern.sh > zzz_filtered-cured-50k-iter-split-sample_data_prop.log 2>&1
# bash run_iter_pattern.sh > zzz_filtered-cured-50k-iter-split-global_data_prop-shuffle.log 2>&1
# bash run_iter_pattern.sh > zzz_llama3b-combine-case.log 2>&1