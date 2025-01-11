# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/lora_merged_reference_model"
reference_model="meta-llama/Llama-3.1-8B-Instruct"
token_select_pattern="all_token_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

# Data and training parameters
# train_data_tag="filtered-cured-50k_all_reference"
# train_data_tag="filtered-cured-50k_all_test"
# train_data_tag="random_subset_50k_all_reference"

train_data_tag="full_dataset"


data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29509

##########
train_data="selected_data/${train_data_tag}.json"
cur_train_model=$base_model

# #### Run calculate_loss.sh script for base model
# echo "start calculating loss for model: ${cur_train_model}"
# BATCH_SIZE_PER_GPU=3
# bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

# # # Run calculate_loss.sh script for reference model
# echo "start calculating loss for reference model: ${reference_model}"
# BATCH_SIZE_PER_GPU=2
# bash_src/calculate_loss.sh "$reference_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

# ## Run Python script to generate data
# echo "start generating labels.."
# python open_instruct/generate_token_label.py \
#     --base_model_name_or_path $cur_train_model \
#     --ref_model_name_or_path $reference_model \
#     --train_data $train_data \
#     --data_prop $data_prop \
#     --select_token_level $select_token_level 


# Define paths for finetuning
BATCH_SIZE_PER_GPU=3
echo "start finetuning..."
bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"


# nohup bash run_reference_model_new.sh > zzz_llama_3_8b_random_subset.log &
# nohup bash run_reference_model_new.sh > zzz_alpaca_eval_52k_all_refenrece.log &