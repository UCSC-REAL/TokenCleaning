# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"


# reference_model="meta-llama/Llama-3.1-8B-Instruct"
reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama8b"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-mistral"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama3b"

with_prompt_token=False
token_select_pattern="semi_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"
select_token_level=global 
# select_token_level=global  ## token_ranking_sample_select global global-positive sample-positive sample union intersection  additional_two_tokens  combine_loss

# Data and training parameters
# train_data_tag="filtered-cured-50k-shuffle-rho-baseline"

# train_data_tag="filtered-cured-50k-rho-baseline-llama8b"
# train_data_tag="filtered-cured-50k-rho-baseline-mistral"
# train_data_tag="filtered-cured-50k-rho-baseline"
# train_data_tag="filtered-cured-50k-rho-baseline-with-prompt"

# train_data_tag="filtered-cured-50k-rho-baseline-with-prompt-llama8b"

train_data_tag="filtered-cured-50k-rho-baseline-llama8b-global"

data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29509


##########
train_data="selected_data/${train_data_tag}.json"

cp "selected_data/filtered-cured-50k_dataset.json" $train_data

cur_train_model=$base_model

# #### Run calculate_loss.sh script for base model
echo "start calculating loss for model: ${cur_train_model}"
BATCH_SIZE_PER_GPU=6
bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"

### Run calculate_loss.sh script for reference model
echo "start calculating loss for reference model: ${reference_model}"
BATCH_SIZE_PER_GPU=6
bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" "$with_prompt_token"

# ## Run Python script to generate data
echo "start generating labels.."
python open_instruct/generate_token_label.py \
    --base_model_name_or_path $cur_train_model \
    --ref_model_name_or_path $reference_model \
    --train_data $train_data \
    --data_prop $data_prop \
    --select_token_level $select_token_level \
    --with_prompt_token $with_prompt_token

# Define paths for finetuning
BATCH_SIZE_PER_GPU=5
echo "start finetuning..."
bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token"


# nohup bash run_rho_baseline.sh > zzz_filtered-cured-50k-rho-baseline.log &
# bash run_rho_baseline.sh > zzz_filtered-cured-50k-rho-baseline-0.3.log 2>&1
# bash run_rho_baseline.sh > zzz_filtered-cured-50k-rho-baseline-llama8b.log 2>&1
# bash run_rho_baseline.sh > zzz_filtered-cured-50k-rho-baseline-mistral.log 2>&1
# bash run_rho_baseline.sh > zzz_filtered-cured-50k-rho-baseline-llama3b.log 2>&1