# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"

token_select_pattern="token_cleaning" #'random'
select_token_level=global 
root_data_path="selected_data"
max_seq_length=2048
main_process_port=29509
data_prop=0.6
random_seed=42
BATCH_SIZE_PER_GPU=6

model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}

reference_model="${model_path}/lora_merged_ds2-10k-warmup-mistral"


train_data_tag="ds2-50k-fixed-model-cleaning-${random_seed}"

##########
train_data="${root_data_path}/${train_data_tag}.json"
cur_train_model=$base_model

cp "${root_data_path}/ds2-50k.json" $train_data

# #### Run calculate_loss.sh script for base model
echo "start calculating loss for model: ${cur_train_model}"
bash_src/calculate_loss.sh "$cur_train_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" 

### Run calculate_loss.sh script for reference model
echo "start calculating loss for reference model: ${reference_model}"
bash_src/calculate_loss.sh "$reference_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS"

# ## Run Python script to generate data
echo "start generating labels.."
python scripts/generate_token_label.py \
    --base_model_name_or_path $cur_train_model \
    --ref_model_name_or_path $reference_model \
    --train_data $train_data \
    --data_prop $data_prop \
    --select_token_level $select_token_level

# Define paths for finetuning
bash_src/finetune.sh "$cur_train_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern"  "$random_seed"

