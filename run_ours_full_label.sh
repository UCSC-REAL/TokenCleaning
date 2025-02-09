# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

select_token_level=global ## global global-positive sample-positive sample union intersection  additional_two_tokens  combine_loss
token_select_pattern=semi_select #"semi_combine_global_half_positive_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

# Data and training parameters
# train_data_tag="filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b_all"
# train_data_tag="filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral_all"

if [[ "$base_model" == *"Mistral-7B-v0.3"* ]]; then
    base_model_tag=mistral
    BATCH_SIZE_PER_GPU=10
elif [[ "$base_model" == *"Llama-3.2-3B"* ]]; then
    base_model_tag=llama3b
    BATCH_SIZE_PER_GPU=6
elif [[ "$base_model" == *"Llama-3.1-8B"* ]]; then
    base_model_tag=llama8b
    BATCH_SIZE_PER_GPU=6
fi

train_data_tag="filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model_all"

data_prop=0.6
max_seq_length=2048
main_process_port=29509

##########
train_data="selected_data/${train_data_tag}.json"

##start from warmup model
cur_train_model="${cluster_root_path}/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-${base_model_tag}"


# cp selected_data/filtered-cured-50k_dataset.json $train_data

# # Define paths for finetuning
BATCH_SIZE_PER_GPU=6
echo "start finetuning..."
bash_src/finetune.sh "$cur_train_model" "$train_data" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"


# bash run_ours_full_label.sh > zzz_ours_full_labels_llama8b.log 2>&1
# bash run_ours_full_label.sh > zzz_ours_full_labels_mistral.log 2>&1