# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path=YOUR_ROOT_PATH
root_data_path="raw_data"

base_model="meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
token_select_pattern=random 
random_seed=42
data_prop=0.6
BATCH_SIZE_PER_GPU=6
model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}

train_data_tag="ds2-50k-random"
train_data="${root_data_path}/${train_data_tag}.json"

#### create dataset replicas
cp "${root_data_path}/ds2-50k.json" $train_data

# finetune 
bash_src/finetune.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed"



