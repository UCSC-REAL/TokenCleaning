#!/bin/bash


# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
# NUM_GPUS=7

export CUDA_VISIBLE_DEVICES=6
NUM_GPUS=1

BATCH_SIZE_PER_GPU=1

######################################
############ base_models #############
declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="8 1 128" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
base_models["meta-llama/Llama-3.2-3B"]="$(($BATCH_SIZE_PER_GPU*$NUM_GPUS)) ${BATCH_SIZE_PER_GPU} 2048"

train_dataset_name='filtered-cured'


model_types=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3")
data_types=( "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")


length=${#model_types[@]}

data_prop=0.4 #0.3
main_process_port=29502


echo "main_process_port: ${main_process_port}"

#############################################################
######## model finetuning on selected training data ######### 
#############################################################

# cluster_root_path="output" ## . for local
cluster_root_path="/mnt/data1/jinlong/token_selection_output"
mkdir -p $cluster_root_path



for base_model in "${!base_models[@]}"
do
    IFS=' ' read -r -a params <<< "${base_models[$base_model]}"
    TOTAL_BATCH_SIZE=${params[0]}
    BATCH_SIZE_PER_GPU=${params[1]}
    max_seq_length=${params[2]}

    for i in "${!model_types[@]}"
    do
        BATCH_SIZE_PER_GPU=${params[1]}
        model_type="${model_types[i]}"
        data_type="${data_types[i]}"

        echo "###### Processing model type:: ${model_type}"

        if [[ $model_type == "base" ]]; then
            model_name_or_path=$base_model
        else
            model_name_or_path=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${model_type}/
        fi

        mkdir -p $cluster_root_path/models/
        # train_data="selected_data/${data_type}.json"
        # echo "train_data:: ${train_data}"

        ### finetune ###
        echo "######### Start finetuning...."
        bash_src/finetune.sh "$model_name_or_path" "$model_type" "$data_type" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$TOTAL_BATCH_SIZE" "$cluster_root_path" "$data_prop" "$main_process_port"

        if [[ $i -eq $((length - 1)) ]]; then
            echo "Reach the last one model --- Finished!"
            continue
        else
            #### calculate the loss for new finetuned_model ####
            new_data_type="${data_types[i+1]}"
            new_model_type=$data_type
            echo "#### start calculating loss ### model pair: (${model_type}, ${new_model_type}) on new subset: ${new_data_type}"

            ##### original model ####
            bash_src/calculate_loss.sh "$model_name_or_path" "$model_type"  "$new_data_type" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port" 

            #### new model ###
            model_name_or_path=$cluster_root_path/models/${base_model}/data_prop_${data_prop}/lora_merged_${new_model_type}/
            bash_src/calculate_loss.sh "$model_name_or_path" "$new_model_type" "$new_data_type" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$main_process_port"

            ## generate data ###
            echo "starting generate data..."
            python open_instruct/generate_data.py --base_model $base_model --model_type $model_type --new_model_type $new_model_type --data_type $new_data_type --data_prop $data_prop --sample_level_top_k_indices True
        fi
    done
done

