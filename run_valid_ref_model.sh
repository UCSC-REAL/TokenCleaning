# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
reference_model="meta-llama/Llama-3.1-8B-Instruct"
token_select_pattern="all_token_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

# Data and training parameters
train_data_tag="valid_samples_all"
data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29509

##########
# 500 validation examples: valid_dataset/valid_examples_all.json

train_data="valid_dataset/${train_data_tag}.json"
cur_train_model=$base_model

# Define paths for finetuning
BATCH_SIZE_PER_GPU=6
echo "start finetuning..."
bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"

