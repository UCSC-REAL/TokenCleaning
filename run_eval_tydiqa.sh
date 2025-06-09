export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8

start_time=$(date +%s)

MODEL=hf #hf

### model path
# model_path=/home/jlpang/LLM_token_selection/output/models/meta-llama/Llama-3.2-3B

# TASK_LISTS=('mmlu' 'bbh' 'gsm8k' "truthfulqa" "hellaswag" "arc_challenge" "piqa"  "openbookqa" 'sciq' 'arc_easy' 'logiqa' 'boolq' 'winogrande') ##task
# TASK_LISTS=('mmlu' 'bbh' 'gsm8k' "truthfulqa" "hellaswag")


#### num_fewshot, batch_size, max_examples(less 1 means proportion)
declare -A TASK_PARAMS=(
    ["mmlu"]="5 16 0.99"
    ["bbh"]="3 64 40"
    ["gsm8k"]="8 48 200"
    ["truthfulqa"]="0 128 0.99"
    ["arc_challenge"]="0 32 0.99"
    ["piqa"]="0 32 0.99"
    ["hellaswag"]="0 128 0.99"
    ["openbookqa"]="0 32 0.99"
    ["sciq"]="0 32 0.99"
    ["arc_easy"]="0 32 0.99"
    ["logiqa"]="0 32 0.99"
    ["boolq"]="0 32 0.99"
    ["winogrande"]="0 32 0.99"
)

####### llama-3.1-8b #####
declare -A TASK_PARAMS=(
    ["mmlu"]="5 8 0.99"
    ["bbh"]="3 32 40"
    ["gsm8k"]="8 48 200"
    ["truthfulqa"]="0 128 0.99"
    ["arc_challenge"]="0 32 0.99"
    ["piqa"]="0 32 0.99"
    ["hellaswag"]="0 128 0.99"
    ["openbookqa"]="0 32 0.99"
    ["sciq"]="0 32 0.99"
    ["arc_easy"]="0 32 0.99"
    ["logiqa"]="0 32 0.99"
    ["boolq"]="0 32 0.99"
    ["winogrande"]="0 32 0.99"
)


# base_model=meta-llama/Llama-3.2-3B
# base_model="meta-llama/Llama-3.1-8B-Instruct"
base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

Train_DATASET_LIST=(
    "filtered-cured-50k-iter-split-global_data_prop_0.6_mistral-non-filtered-fixed-base-model_4"
    ) 

# data_prop_list=(0.6) # 0.3
data_prop=0.6

# TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')
TASK_LISTS=()


for idx in "${!Train_DATASET_LIST[@]}"; do
    train_dataset_name="${Train_DATASET_LIST[$idx]}"
    echo "Index: $idx, Dataset: $train_dataset_name"

    echo "##### train_dataset_name: ${train_dataset_name}"

    if [[ "$train_dataset_name" == *"llama3b"* ]]; then
        base_model=meta-llama/Llama-3.2-3B
    elif [[ "$train_dataset_name" == *"llama8b"* ]]; then
        base_model="meta-llama/Llama-3.1-8B"
    elif [[ "$train_dataset_name" == *"mistral"* ]]; then
        base_model="mistralai/Mistral-7B-v0.3"
    else
        base_model=$base_model
    fi

    ##############################################################
    model_path="/mnt/data1/jinlong/token_selection_output/models/${base_model}"



    if [[ "$train_dataset_name" == *"llama3b"* ]]; then
        declare -A TASK_PARAMS=(
            ["mmlu"]="5 16 0.99"
            ["bbh"]="3 64 40"
            ["gsm8k"]="8 48 200"
            ["truthfulqa"]="0 128 0.99"
            ["arc_challenge"]="0 32 0.99"
            ["piqa"]="0 32 0.99"
            ["hellaswag"]="0 128 0.99"
            ["openbookqa"]="0 32 0.99"
            ["sciq"]="0 32 0.99"
            ["arc_easy"]="0 32 0.99"
            ["logiqa"]="0 32 0.99"
            ["boolq"]="0 32 0.99"
            ["winogrande"]="0 32 0.99"
            ["squadv2"]="0 64 0.99"
            ["squad_completion"]="0 64 0.99"
            ["triviaqa"]="0 64 0.99"
            ["humaneval"]="0 64 0.99"
        )
    else
        declare -A TASK_PARAMS=(
            ["mmlu"]="5 8 0.99"
            ["bbh"]="3 32 40"
            ["gsm8k"]="8 48 200"
            ["truthfulqa"]="0 128 0.99"
            ["arc_challenge"]="0 32 0.99"
            ["piqa"]="0 32 0.99"
            ["hellaswag"]="0 128 0.99"
            ["openbookqa"]="0 32 0.99"
            ["sciq"]="0 32 0.99"
            ["arc_easy"]="0 32 0.99"
            ["logiqa"]="0 32 0.99"
            ["boolq"]="0 32 0.99"
            ["winogrande"]="0 32 0.99"
            ["squadv2"]="0 64 0.99"
            ["squad_completion"]="0 64 0.99"
            ["triviaqa"]="0 64 0.99"
            ["humaneval"]="0 64 0.99"

        )
    fi

    echo "*** data_prop_list: ${data_prop_list} ***"
    echo "*** base model: ${base_model} ***"
    echo "*** current data prop value: ${data_prop} ***"
    if [[ $train_dataset_name == 'base' ]]; then
        pretrained_model=$base_model
    else
        pretrained_model=${model_path}/data_prop_${data_prop}/lora_merged_${train_dataset_name}
    fi

    echo "######## evaluation model: ${train_dataset_name} #############"

    OUTPUT_PATH=token_selection_results/${data_prop}/${train_dataset_name}

    mkdir -p $OUTPUT_PATH

    ###########################################
    #### tydiqa eval ####
    ###########################################
    CUDA_VISIBLE_DEVICES=$idx python -m eval.tydiqa.run_eval \
        --data_dir raw_data/eval/tydiqa/ \
        --n_shot 1 \
        --max_num_examples_per_lang 100 \
        --max_context_length 512 \
        --save_dir $OUTPUT_PATH \
        --model_name_or_path $pretrained_model \
        --tokenizer_name_or_path $pretrained_model \
        --eval_batch_size 5 &
        # --use_vllm

done 

echo "all experiments finished!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"



# bash run_eval_token_selection.sh > zzz_iter_global_combine_active_split.log 2>&1
# bash run_eval_token_selection.sh > zzz_iter_global_combine_active_split.log 2>&1
# bash run_eval_token_selection.sh > zzz_iter_global_combine_two_types_labels.log 2>&1
# bash run_eval_token_selection.sh > zzz_zzzz_llama8b.log 2>&1
# bash run_eval_token_selection.sh > zzz_zzzz_mistral.log 2>&1

