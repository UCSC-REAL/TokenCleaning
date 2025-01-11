# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8

# Define model paths and tags
base_model="meta-llama/Llama-3.2-3B"

# reference_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/lora_merged_reference_model"
reference_model="meta-llama/Llama-3.1-8B-Instruct"
base_model_tag="llama-3.2-3B-base"
reference_model_tag="reference_model"  # Make sure this is set correctly

token_select_pattern="semi_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"

# Data and training parameters
train_data_tag="filtered-cured-50k_all_reference"
# train_data_tag="filtered-cured-50k_all_test"
data_prop=0.6
max_seq_length=2048
BATCH_SIZE_PER_GPU=3
main_process_port=29509



# Run calculate_loss.sh script for base model
echo "start calculating loss for base model: ${base_model}"
bash_src/calculate_loss.sh "$base_model" "$base_model_tag" "$train_data_tag" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

# # Run calculate_loss.sh script for reference model
echo "start calculating loss for reference model: ${reference_model}"
BATCH_SIZE_PER_GPU=2
bash_src/calculate_loss.sh "$reference_model" "$reference_model_tag" "$train_data_tag" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

### Run Python script to generate data
echo "start generating labels.."
python open_instruct/generate_data.py \
    --base_model $base_model \
    --model_type $base_model_tag \
    --new_model_type $reference_model_tag \
    --data_type $train_data_tag \
    --data_prop $data_prop \
    --sample_level_top_k_indices True 

# Define paths for finetuning
cluster_root_path="/data1/jinlong/token_selection_output"
finetune_model_tag="reference_based_selected"
BATCH_SIZE_PER_GPU=2
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NUM_GPUS))

# Run finetune.sh script
echo "start finetuning..."
model_name_or_path=$base_model
bash_src/finetune.sh "$model_name_or_path" "$finetune_model_tag" "$train_data_tag" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$TOTAL_BATCH_SIZE" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"


# bash run_reference_model.sh > zzz_reference_model_selected_llama_3_8b.log 2>&1
