# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"

# reference_model="meta-llama/Llama-3.1-8B-Instruct"
token_select_pattern="all_token_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

random_seed_list=(41 43)

with_prompt_token=False


data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29509
##########

for random_seed in "${random_seed_list[@]}"; do
    # Data and training parameters
    # train_data_tag="filtered-cured-10k-shuffle-warmup" ## for warmup model
    # train_data_tag="filtered-cured-10k-warmup" ## for warmup model
    train_data_tag="filtered-cured-50k-full-baseline-mistral-${random_seed}"
    train_data="selected_data/${train_data_tag}.json"
    cur_train_model=$base_model

    cp "selected_data/filtered-cured-50k_dataset.json" $train_data

    # Define paths for finetuning
    BATCH_SIZE_PER_GPU=6
    echo "start finetuning..."
    bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"

done

# bash run_full_baseline.sh > zzz_filtered-cured-50k-shuffle-full-baseline.log 2>&1
# bash run_full_baseline.sh > zzz_filtered-cured-10k-warmup_mistral.log 2>&