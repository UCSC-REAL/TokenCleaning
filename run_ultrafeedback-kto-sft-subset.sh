# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8




cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"
base_model="princeton-nlp/Llama-3-Base-8B-SFT"

# reference_model="meta-llama/Llama-3.1-8B-Instruct"
token_select_pattern="all_token_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

random_seed=42
with_prompt_token=False

# Data and training parameters
# train_data_tag="filtered-cured-10k-shuffle-warmup" ## for warmup model
# train_data_tag="filtered-cured-10k-warmup" ## for warmup model

data_prop=1
max_seq_length=800
main_process_port=29509
cur_train_model=$base_model

##########
train_data_tag="ultrafeedback-sft-identical-pairs-7387"
# train_data_tag="ultrafeedback-sft-identical-pairs-14774"

# train_data_tag="ultrafeedback-sft-identical-pairs-7387-rejected"
train_data="${train_data_tag}.json"

# cp "ultrafeedback-kto-sft-random-identical-subset.json" $train_data
# cp "ultrafeedback-kto-sft-random-identical-subset-chosen-only.json" $train_data
# cp "ultrafeedback-sft-identical-pairs-7387.json" $train_data


# Define paths for finetuning
BATCH_SIZE_PER_GPU=5
echo "start finetuning..."
bash_src/finetune_ultrafeedback_lora.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"

#######################################################
#######################################################

# # Data and training parameters
# train_data_tag="ultrafeedback-kto-sft-random-identical-subset-chosen-only-full"
# train_data="${train_data_tag}.json"

# # cp "ultrafeedback-kto-sft-random-identical-subset.json" $train_data
# cp "ultrafeedback-kto-sft-random-identical-subset-chosen-only.json" $train_data

# BATCH_SIZE_PER_GPU=1
# echo "start finetuning..."
# bash_src/finetune_ultrafeedback_full.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern" "$with_prompt_token" "$random_seed"



# mv lora_merged_ultrafeedback-sft-identical-pairs-7387 /mnt/data1/jinlong/DPO-noisy-outputs/llama-3-8b-sft-identical-pairs-7387
