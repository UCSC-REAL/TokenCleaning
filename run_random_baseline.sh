# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
NUM_GPUS=6


cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"

# reference_model="meta-llama/Llama-3.1-8B-Instruct"
token_select_pattern="random_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

random_seed=43
with_prompt_token=False
# Data and training parameters
train_data_tag="filtered-cured-50k-random-baseline-${random_seed}"


data_prop=0.6
max_seq_length=2048
main_process_port=29509
train_data="selected_data/${train_data_tag}.json"
cur_train_model=$base_model

cp "selected_data/filtered-cured-50k_dataset.json" $train_data

# Define paths for finetuning
BATCH_SIZE_PER_GPU=6
echo "start finetuning..."
bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"


# bash run_random_baseline.sh > zzz_filtered-cured-50k-random-baseline-0.6.log 2>&1

