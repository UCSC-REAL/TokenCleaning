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


base_model=meta-llama/Llama-3.2-3B
# base_model="meta-llama/Llama-3.1-8B-Instruct"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

# Train_DATASET_LIST=("base")
# Train_DATASET_LIST=("filtered-cured-50k-shuffle-random-baseline")
# Train_DATASET_LIST=("filtered-cured-10k-warmup")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline")
# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b")


# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-mistral" "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral")

# Train_DATASET_LIST=("filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b_all")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral_all")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b" "filtered-cured-50k-iter-split-global_data_prop_0.3_llama8b" "filtered-cured-50k-iter-split-global_data_prop_0.3_mistral")

# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-global")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-data-prop-0.6-fixed-base-loss-using-warmup-label_llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered" "filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered-with-prompt" "filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-with-prompt")



# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt-llama8b")


# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6-non-filtered-combine_active-split-data-prop-0.6-fixed-base-loss-using-warmup-label_llama3b")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3-non-filtered-combine_active-split-data-prop-0.6-fixed-base-loss-using-warmup-label_llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3-non-filtered-combine_active-split-data-prop-0.45-fixed-base-loss-using-warmup-label_llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3-non-filtered-combine_active-split-sample-data-prop-0.3-fixed-base-loss-using-warmup-label_llama3b")
# Train_DATASET_LIST=("filtered-cured-50k-iter-global_prop_0.3-non-filtered-combine_warmup-sample-prop-0.3-label_llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-global-prop-0.3-non-filtered-combine-warmup-sample-prop-0.6-label-llama3b")

# Train_DATASET_LIST=("filtered-cured-50k-iter-global-prop-0.6-non-filtered-combine-warmup-sample-prop-0.3-label-llama3b" "filtered-cured-50k-iter-global-prop-0.6-non-filtered-combine-warmup-sample-prop-0.6-label-llama3b")


# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt-llama3b")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-with-prompt-llama3b-new")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.3_llama3b-non-filtered-fixed-base-model")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.7_llama3b-non-filtered-fixed-base-model")

# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-global" "filtered-cured-50k-rho-baseline-sample")
# Train_DATASET_LIST=("base")
# Train_DATASET_LIST=("filtered-cured-10k-warmup-llama3b")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline")

# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.8_llama3b-non-filtered-fixed-base-model")
# Train_DATASET_LIST=("filtered-cured-50k-iter-split-global_data_prop_0.6_llama3b-non-filtered-fixed-base-model_all")

# Train_DATASET_LIST=("filtered-cured-50k-full-baseline" "filtered-cured-50k-random-baseline")

# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-mistral-global" ) #"filtered-cured-50k-rho-baseline-mistral-global"   "filtered-cured-50k-rho-baseline-llama8b-global"



# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-global-llama3b") 
Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-sample-llama3b") 

data_prop_list=(0.3 0.4 0.5 0.7 0.8 0.9)

# TASK_LISTS=('mmlu' 'bbh' 'gsm8k' "truthfulqa" "hellaswag" "arc_challenge" "piqa"  "openbookqa" 'sciq' 'arc_easy' 'logiqa' 'boolq' 'winogrande') ##task
# TASK_LISTS=('mmlu' 'bbh' 'gsm8k' "truthfulqa" "hellaswag")
# TASK_LISTS=("arc_challenge" "piqa"  "openbookqa" 'sciq' 'arc_easy' 'logiqa' 'boolq' 'winogrande') ##task
# TASK_LISTS=( "truthfulqa" "hellaswag" "arc_challenge" "openbookqa" "boolq")

# "squad_completion"
TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')

# TASK_LISTS=("sciq" "triviaqa" "piqa" 'arc_easy' 'logiqa' 'winogrande') #"squadv2"
# TASK_LISTS=('bbh' 'gsm8k') #"squadv2"


# TASK_LISTS=('mmlu' 'bbh' 'gsm8k' "truthfulqa" "hellaswag" "arc_challenge" "openbookqa" "boolq" "sciq" "triviaqa" "piqa" 'arc_easy' 'logiqa' 'winogrande') #"squadv2"



for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
    echo "##### train_dataset_name: ${train_dataset_name}"
    # if [[ "$train_dataset_name" == *"0.3"* ]]; then
    #     data_prop_list=(0.3)
    # elif [[ "$train_dataset_name" == *"0.6"* ]]; then
    #     data_prop_list=(0.6)
    # fi

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
    # model_tags=("${train_dataset_name}_0" "${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    # model_tags=("${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3")
    # model_tags=("${train_dataset_name}_1" "${train_dataset_name}_2" "${train_dataset_name}_3" "${train_dataset_name}_4")
    # model_tags=("${train_dataset_name}_3")
    model_tags=("${train_dataset_name}")
    ##############################################################


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

    for data_prop in ${data_prop_list[@]}; do
        echo "*** current data prop value: ${data_prop} ***"
        
        for model_tag in ${model_tags[@]}; do

            if [[ $model_tag == 'base' ]]; then
                pretrained_model=$base_model
            else
                pretrained_model=${model_path}/data_prop_${data_prop}/lora_merged_${model_tag}
            fi

            echo "######## evaluation model: ${model_tag} #############"

            OUTPUT_PATH=token_selection_results/${data_prop}/${model_tag}

            mkdir -p $OUTPUT_PATH

            declare -A MODEL_ARGS_PARAMS=(
                ["mmlu"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["bbh"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["gsm8k"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["truthfulqa"]="pretrained=${pretrained_model},dtype=bfloat16"  #,load_in_8bit=True
                ["arc_challenge"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["piqa"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["hellaswag"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["openbookqa"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["sciq"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["arc_easy"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["logiqa"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["boolq"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["winogrande"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["squadv2"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["triviaqa"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["squad_completion"]="pretrained=${pretrained_model},dtype=bfloat16"
                ["humaneval"]="pretrained=${pretrained_model},dtype=bfloat16"

            )


            for idx in "${!TASK_LISTS[@]}"; do

                task=${TASK_LISTS[$idx]}
                params=(${TASK_PARAMS[$task]})  # splits
                num_fewshot=${params[0]}
                batch_size=${params[1]}
                max_examples_per_task=${params[2]}
                gpu_idx=$((idx % 8))
                model_args=${MODEL_ARGS_PARAMS[$task]}

                echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

                accelerate launch --multi-gpu --main_process_port 29519 --num_processes $NUM_GPUs \
                        -m lm_eval --model $MODEL \
                        --model_args $model_args \
                        --tasks $task \
                        --batch_size $batch_size \
                        --num_fewshot $num_fewshot \
                        --limit $max_examples_per_task \
                        --output_path $OUTPUT_PATH \
                        --seed 42 \
                        --trust_remote_code
                        # --device cuda
                        
                sleep 3s

            done

            ###########################################
            #### tydiqa eval ####
            ###########################################
            CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
                --data_dir raw_data/eval/tydiqa/ \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir $OUTPUT_PATH \
                --model_name_or_path $pretrained_model \
                --tokenizer_name_or_path $pretrained_model \
                --eval_batch_size 60

        done
    done
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

