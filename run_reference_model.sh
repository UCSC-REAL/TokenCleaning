# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"

token_select_pattern=default 

random_seed=42
with_prompt_token=False
data_prop=0.6

model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}


# Data and training parameters
train_data_tag="filtered-cured-10k-warmup-mistral-new-${random_seed}"
train_data="selected_data/${train_data_tag}.json"
cp "selected_data/filtered-cured-10k-warmup.json" $train_data

BATCH_SIZE_PER_GPU=6
echo "start finetuning..."
bash_src/finetune.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed"

