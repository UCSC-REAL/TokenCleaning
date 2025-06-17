# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


start_time=$(date +%s)

#### basic config
max_seq_length=2048
BATCH_SIZE_PER_GPU=6
cluster_root_path="/mnt/data1/jinlong/token_selection_output"
root_data_path="selected_data"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

select_token_level=global # sample
token_select_pattern=token_cleaning #random default
data_prop=0.6
subset_size=5

model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}

### training data
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered-with-prompt" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-with-prompt")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model")

Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_mistral-non-filtered-fixed-base-model")




for train_dataset_name in ${Train_DATASET_LIST[@]}; do

    echo "*** current train dataset name: ${train_dataset_name} ***"

    echo "*** subset json file generation ***"
    python scripts/generate_subset.py --generate_train_data_name $train_dataset_name --subset_size $subset_size

    ## Subset-level Iteration ##
    for idx in $(seq 0 $((subset_size - 1))); do
        train_data_tag="${train_dataset_name}_${idx}"
        train_data="${root_data_path}/${train_data_tag}.json"

        ## first-round reference model
        if [[ $idx -eq 0 ]]; then
            warmup_token_select_pattern="default"
            bash_src/finetune.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop"  "$warmup_token_select_pattern" "$random_seed"

        else
            ref_model=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${train_dataset_name}_$((idx-1))

            # Compute token loss 
            bash_src/calculate_loss.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"     
            bash_src/calculate_loss.sh "$ref_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 

            ## Run Python script to generate data
            echo "start generating labels.."
            python scripts/generate_token_label.py \
                --base_model_name_or_path $base_model  \
                --ref_model_name_or_path $ref_model \
                --train_data $train_data \
                --data_prop $data_prop \
                --select_token_level $select_token_level

            # Run finetune.sh script
            echo "start finetuning..."
            bash_src/finetune.sh "$ref_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern"  "$random_seed"

        fi
    done 

done


