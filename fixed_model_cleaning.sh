# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"


# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama3b"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama8b"
reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-mistral"

with_prompt_token=False
token_select_pattern="token_cleaning" #'random'
select_token_level=global 

max_seq_length=2048
main_process_port=29509
data_prop_list=(0.6)

random_seed=42


train_data_tag="filtered-cured-50k-fixed-model-cleaning-${random_seed}"

##########
train_data="selected_data/${train_data_tag}.json"
cur_train_model=$base_model

cp "selected_data/filtered-cured-50k_dataset.json" $train_data

# #### Run calculate_loss.sh script for base model
echo "start calculating loss for model: ${cur_train_model}"
BATCH_SIZE_PER_GPU=6
bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"

### Run calculate_loss.sh script for reference model
echo "start calculating loss for reference model: ${reference_model}"
BATCH_SIZE_PER_GPU=6
bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"

for data_prop in ${data_prop_list[@]}; do
    echo "*** data prop: ${data_prop} ***"

    # ## Run Python script to generate data
    echo "start generating labels.."
    python scripts/generate_token_label.py \
        --base_model_name_or_path $cur_train_model \
        --ref_model_name_or_path $reference_model \
        --train_data $train_data \
        --data_prop $data_prop \
        --select_token_level $select_token_level \
        --with_prompt_token $with_prompt_token

    # Define paths for finetuning
    BATCH_SIZE_PER_GPU=6
    echo "start finetuning..."
    bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"

done
