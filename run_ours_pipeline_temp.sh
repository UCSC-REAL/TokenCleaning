# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8


start_time=$(date +%s)

cluster_root_path="/mnt/data1/jinlong/token_selection_output"

########## Define model paths and tags ##############
base_model="meta-llama/Llama-3.2-3B"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"


########## warmup-label setting ###################
select_token_level=global-half-positive ## token_ranking_sample_select global global-positive sample-positive sample union intersection  additional_two_tokens  combine_loss
token_select_pattern="semi_select" #'random_semi_shift', 'semi_select', 'random_select', "loss_ranking_select", "all_token_select"
warmup_data_prop=0.3 # if means nothing when we use the global-positive or sample-positive

######################################
# Note: iter_train_data_name: fixed (should be check the existence)
# warmup_train_dataset_name can be changed (not important just for temperate store)
# combine_train_data_tag can be changed (the file name used for final finetuning)
######################################

##### llama-3b ###
warmup_train_dataset_name="filtered-cured-50k-active-split-sample-data-prop-0.3-fixed-base-loss-using-warmup-llama3b" 
# iter_train_data_name="filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b"
# iter_train_data_name="filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered"
# iter_train_data_name="filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered"
iter_train_data_name="filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model" ##fixed base model
# combine_train_data_name="filtered-cured-50k-iter-split-global_data_prop_0.3-non-filtered-combine_active-split-sample-data-prop-0.3-fixed-base-loss-using-warmup-label_llama3b"
# combine_train_data_name="filtered-cured-50k-iter-global-prop-0.3-non-filtered-combine-warmup-sample-prop-0.6-label-llama3b"
# combine_train_data_name="filtered-cured-50k-iter-global-prop-0.6-non-filtered-combine-warmup-sample-prop-0.3-label-llama3b"

combine_train_data_name="filtered-cured-50k-iter-global-prop-0.6-non-filtered-fixed-combine-warmup-global-half-positive-label-llama3b"


# ## llama-8b
# iter_train_data_name = "filtered-cured-50k-iter-split-global_data_prop_0.3_llama8b"
# warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b"
# combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b"


### mistral-7b
# iter_train_data_name = "filtered-cured-50k-iter-split-global_data_prop_0.3_mistral"
# warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-mistral"
# combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral"



#################################################################################
if [[ "$base_model" == *"Mistral-7B-v0.3"* ]]; then
    base_model_tag=mistral
    BATCH_SIZE_PER_GPU=10
elif [[ "$base_model" == *"Llama-3.2-3B"* ]]; then
    base_model_tag=llama3b
    BATCH_SIZE_PER_GPU=6
elif [[ "$base_model" == *"Llama-3.1-8B"* ]]; then
    base_model_tag=llama8b
    BATCH_SIZE_PER_GPU=6

fi

reference_model="${cluster_root_path}/models/${base_model}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-${base_model_tag}"

################################################################################
##### split orginal token loss prepration ###########
################################################################################
echo "*** warmup loss spliting ***"
python open_instruct/token_loss_split.py --root_path $cluster_root_path --base_model_name_or_path $base_model --train_dataset_name $warmup_train_dataset_name

#################################################################################
########## generate subset json file for finetuning ######
#################################################################################
echo "*** subset json file generation ***"

python open_instruct/generate_subset.py --generate_train_data_name $warmup_train_dataset_name
python open_instruct/generate_subset.py --generate_train_data_name $combine_train_data_name


#################################################################################
########## generate wamrup label #######
#################################################################################
echo "*** warmup label generation ***"
echo "*** current data prop value: ${warmup_data_prop} ***"

train_data_tag_list=("${warmup_train_dataset_name}_0" "${warmup_train_dataset_name}_1" "${warmup_train_dataset_name}_2" "${warmup_train_dataset_name}_3" "${warmup_train_dataset_name}_4")

for idx in "${!train_data_tag_list[@]}"; do
    train_data_tag=${train_data_tag_list[${idx}]}
    train_data="selected_data/${train_data_tag}.json"

    if [[ $idx -eq 0 ]]; then
        cur_train_model=$base_model
    else
        cur_train_model=$cluster_root_path/models/${base_model}/data_prop_${warmup_data_prop}/lora_merged_${train_data_tag_list[$((idx-1))]}
    fi
    ## Run Python script to generate data
    echo "start generating labels.."
    python open_instruct/generate_token_label.py \
        --base_model_name_or_path $cur_train_model \
        --ref_model_name_or_path $reference_model \
        --train_data $train_data \
        --data_prop $warmup_data_prop \
        --select_token_level $select_token_level \
        --subset_idx $idx \
        --num_subset "${#train_data_tag_list[@]}"
done


#################################################################################
########## combine labels for finetuning ######
#################################################################################
echo "*** combining labels ***"

python open_instruct/combine_labels.py --iter_train_data_tag $iter_train_data_name --warmup_train_data_tag $warmup_train_dataset_name --combine_train_data_tag $combine_train_data_name


#################################################################################
########## start finetuning ######
#################################################################################
echo "*** start finetuning for combine setting ***"
data_prop=0.6
bash bash_src/run_fixed_label_finetune.sh "$base_model" "$token_select_pattern" "$combine_train_data_name" "$data_prop" "$BATCH_SIZE_PER_GPU" "$reference_model"





end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $minutes minutes"


# bash run_ours_pipeline.sh  > zzz_pipline_llama3b.log 2>&1