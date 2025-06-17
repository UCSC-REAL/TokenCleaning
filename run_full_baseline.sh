# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


cluster_root_path="/mnt/data1/jinlong/token_selection_output"
root_data_path="selected_data"
# Define model paths and tags
# base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
base_model="mistralai/Mistral-7B-v0.3"

token_select_pattern="token_cleaning" 

random_seed_list=(41 43)

with_prompt_token=False
data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=6
main_process_port=29509



for random_seed in "${random_seed_list[@]}"; do

    train_data_tag="ds2-50k-full-baseline-mistral-${random_seed}"
    train_data="${root_data_path}/${train_data_tag}.json"
    cur_train_model=$base_model

    cp "${root_data_path}/ds2-50k.json" $train_data

    echo "start finetuning..."
    bash_src/finetune.sh "$cur_train_model" "$train_data" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$model_path" "$data_prop" "$token_select_pattern" "$random_seed"

done

