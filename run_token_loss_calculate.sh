# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"


# reference_model="meta-llama/Llama-3.1-8B-Instruct"
# 10k warm-up model
# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/data_prop_0.6/lora_merged_filtered-cured-10k-shuffle-warmup"
reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama3b"
#full referencem model
# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/data_prop_0.6/lora_merged_filtered-cured-50k-full-baseline"



max_seq_length=2048
BATCH_SIZE_PER_GPU=6
main_process_port=29509


# Data and training parameters
# train_data_tag="filtered-cured-50k-shuffle-rho-baseline"
train_data_tag="filtered-cured-50k-rho-baseline"

##########
train_data="selected_data/${train_data_tag}.json"
cur_train_model=$base_model

# #### Run calculate_loss.sh script for base model
# echo "start calculating loss for model: ${cur_train_model}"
# BATCH_SIZE_PER_GPU=10
# bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

# # # Run calculate_loss.sh script for reference model
echo "start calculating loss for reference model: ${reference_model}"
BATCH_SIZE_PER_GPU=6
bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"
