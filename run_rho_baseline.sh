# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama8b"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-mistral"
# reference_model="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-llama3b"

token_select_pattern="token_cleaning" #'random'
select_token_level=global #sample

# Data and training parameters
# train_data_tag="filtered-cured-50k-rho-baseline-llama8b"
# train_data_tag="filtered-cured-50k-rho-baseline-mistral"

# train_data_tag="filtered-cured-50k-rho-baseline-with-prompt-llama8b"
train_data_tag="filtered-cured-50k-rho-baseline-llama3b-global-ref-8binst"
data_prop=0.6
BATCH_SIZE_PER_GPU=3

model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}

##########
train_data="selected_data/${train_data_tag}.json"

cp "selected_data/filtered-cured-50k_dataset.json" $train_data

# #### Run calculate_loss.sh script for base model
# echo "start calculating loss for model: ${base_model}"
# BATCH_SIZE_PER_GPU=6
# bash_src/calculate_loss.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 

# ### Run calculate_loss.sh script for reference model
# echo "start calculating loss for reference model: ${reference_model}"
# BATCH_SIZE_PER_GPU=6
# bash_src/calculate_loss.sh "$reference_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 

# ## Run Python script to generate data
echo "start generating labels.."
python scripts/generate_token_label.py \
    --base_model_name_or_path $base_model \
    --ref_model_name_or_path $reference_model \
    --train_data $train_data \
    --data_prop $data_prop \
    --select_token_level $select_token_level

# Define paths for finetuning
BATCH_SIZE_PER_GPU=6
echo "start finetuning..."
bash_src/finetune.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed"

