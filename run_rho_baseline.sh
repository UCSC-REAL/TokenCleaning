# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path=YOUR_ROOT_PATH
root_data_path="raw_data"

base_model="meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
token_select_pattern="token_cleaning" #'random'
select_token_level=global #sample
data_prop=0.6
BATCH_SIZE_PER_GPU=6
model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}

## reference model
ref_model="${model_path}/lora_merged_ds2-10k-warmup"
train_data_tag="ds2-50k-rho"


#### create dataset replicas
train_data="${root_data_path}/${train_data_tag}.json"
cp "${root_data_path}/ds2-50k.json" $train_data

# Compute token loss
bash_src/calculate_loss.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 
bash_src/calculate_loss.sh "$ref_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 

## Generate token label
python scripts/generate_token_label.py \
    --base_model_name_or_path $base_model \
    --ref_model_name_or_path $ref_model \
    --tokenizer_name_or_path $base_model \
    --train_data $train_data \
    --data_prop $data_prop \
    --select_token_level $select_token_level

# finetune
bash_src/finetune.sh "$base_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed"

