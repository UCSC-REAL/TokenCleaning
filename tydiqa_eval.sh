export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

start_time=$(date +%s)


# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_other_two_types_label") 
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_combine_other_two_types_label")
# Train_DATASET_LIST=("filtered-cured-10k-shuffle-warmup")

# Train_DATASET_LIST=("filtered-cured-50k-shuffle-iter-split-global_data_prop_0.3")
# Train_DATASET_LIST=("filtered-cured-50k-shuffle-iter-split-global_data_prop_0.6")
# Train_DATASET_LIST=("filtered-cured-50k-shuffle-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label")

###10k warmup model
# base_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/data_prop_0.6/lora_merged_filtered-cured-10k-full-model"
# base_model="/mnt/data1/jinlong/token_selection_output/models/meta-llama/Llama-3.2-3B/data_prop_0.6/lora_merged_valid_samples_all/"


base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

# Train_DATASET_LIST=("filtered-cured-50k-shuffle-full-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-shuffle-random-baseline")
# Train_DATASET_LIST=("filtered-cured-10k-warmup")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral_all")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b_all")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-mistral")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama3b")

# Train_DATASET_LIST=("base")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama8b")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_mistral")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-global")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-data-prop-0.6-fixed-base-loss-using-warmup-label_llama3b")

Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt")


data_prop=0.6 # 0.3
eval_dataset_name='tydiqa'

model_path="/mnt/data1/jinlong/token_selection_output/models/${base_model}/data_prop_${data_prop}"
MODEL=hf #hf


for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
    echo "##### train_dataset_name: ${train_dataset_name}"

    # model_tags=("${train_dataset_name}_4")
    # model_tags=("${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    # model_tags=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    model_tags=("${train_dataset_name}")

    # for model_tag in ${model_tags[@]}; do
    for idx in ${!model_tags[@]}; do

        model_tag=${model_tags[$idx]} 

        if [[ $model_tag == 'base' ]]; then
            pretrained_model=$base_model
        else
            pretrained_model=${model_path}/lora_merged_${model_tag}
        fi

        echo "######## evaluation model: ${model_tag} #############"

        OUTPUT_PATH=tydiqa_eval_results/${data_prop}/${model_tag}

        mkdir -p $OUTPUT_PATH

        CUDA_VISIBLE_DEVICES=$idx python -m eval.tydiqa.run_eval \
            --data_dir raw_data/eval/tydiqa/ \
            --n_shot 1 \
            --max_num_examples_per_lang 100 \
            --max_context_length 512 \
            --save_dir $OUTPUT_PATH \
            --model_name_or_path $pretrained_model \
            --tokenizer_name_or_path $pretrained_model \
            --eval_batch_size 60 &
    done
    wait
done 

OUTPUT_PATH=tydiqa_eval_results/${data_prop}/

for train_dataset_name in "${Train_DATASET_LIST[@]}"; do

    # model_tags=("${train_dataset_name}_4")
    # model_tags=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    model_tags=("${train_dataset_name}")
    # model_tags=("${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")

    for model_tag in "${model_tags[@]}"; do        
        # echo "###### Processing training dataset :: ${train_dataset_name}"
        # echo "###### Processing model_tag :: ${model_tag}"
        python3 read_results_token.py --root_result_path $OUTPUT_PATH --eval_dataset $eval_dataset_name --train_dataset $train_dataset_name --baseline_tag $model_tag 

    done

done



echo "all experiments finished!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"
