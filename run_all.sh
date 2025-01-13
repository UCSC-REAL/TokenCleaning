#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,7
NUM_GPUS=6



start_time=$(date +%s)

BATCH_SIZE_PER_GPU=2

######################################
############ base_models #############
declare -A base_models
# base_models["meta-llama/Meta-Llama-3.1-8B"]="8 1 128" # TOTAL_BATCH_SIZE BATCH_SIZE_PER_GPU max_seq_length
base_models["meta-llama/Llama-3.2-3B"]="$(($BATCH_SIZE_PER_GPU*$NUM_GPUS)) ${BATCH_SIZE_PER_GPU} 2048"

# Train_DATASET_LIST=('filtered-cured-50k-all-iter-sample-subset-small-new' 'filtered-cured-50k-all-iter-global-subset-small-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-non-iter-sample-subset-new' 'filtered-cured-50k-all-non-iter-global-subset-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-iter-union-subset-small-new') #
# Train_DATASET_LIST=('filtered-cured-50k-all-iter-additional_two_tokens-subset-small-new') #

# Train_DATASET_LIST=('filtered-cured-50k-all-iter-combine-loss-subset-small-new') #

# Train_DATASET_LIST=('filtered-cured-50k-all-non-iter-global') #
# Train_DATASET_LIST=('filtered-cured-50k-non-iter-split-global-new') #
Train_DATASET_LIST=("filtered-cured-50k-non-iter-split-global-new-randtok")
data_prop=0.3 # 0.3
main_process_port=29516


token_select_pattern="semi_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"


for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
    echo "##### train_dataset_name: ${train_dataset_name}"

    # model_types=("base" "${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3"  "${train_dataset_name}_4" "${train_dataset_name}_5" "${train_dataset_name}_6" "${train_dataset_name}_7" "${train_dataset_name}_8")
    # data_types=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4" "${train_dataset_name}_5" "${train_dataset_name}_6" "${train_dataset_name}_7" "${train_dataset_name}_8" "${train_dataset_name}_9")


    model_types=("base" "${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3")
    data_types=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")


    length=${#model_types[@]}

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

            ###finetune ###
            echo "######### Start finetuning...."
            bash_src/finetune.sh "$model_name_or_path" "$model_type" "$data_type" "$max_seq_length" "$BATCH_SIZE_PER_GPU" "$NUM_GPUS" "$base_model" "$TOTAL_BATCH_SIZE" "$cluster_root_path" "$data_prop" "$main_process_port" "$token_select_pattern"


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

                python open_instruct/generate_data.py \
                    --base_model $base_model \
                    --model_type $model_type \
                    --new_model_type $new_model_type \
                    --data_type $new_data_type \
                    --data_prop $data_prop \
                    --select_token_level $global
            fi
        done
    done
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"


# nohup bash run_all.sh > zzz_sample_iter_subset_small_new.log &
# nohup bash run_all.sh > zzz_sample_non_iter_subset_small_new.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-union.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-additional-two-tokens.log &
# nohup bash run_all.sh > zzz_sample_iter_subset_small_new-intersection.log &

#  bash run_all.sh > zzz_filtered-cured-50k-non-iter-split-global.log 2>&1
# bash run_all.sh > zzz_filtered-cured-50k-non-iter-split-global-new.log 2>&1
# bash run_all.sh > zzz_filtered-cured-50k-non-iter-split-global-new-0.6.log 2>&1
# bash run_all.sh > zzz_filtered-cured-50k-non-iter-split-global-new-randtok.log 2>&1