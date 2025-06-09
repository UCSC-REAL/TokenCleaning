export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUs=4


start_time=$(date +%s)

MODEL=hf #hf


base_model=meta-llama/Llama-3.2-3B
# base_model="meta-llama/Llama-3.1-8B-Instruct"
# base_model="meta-llama/Llama-3.1-8B"
# base_model="mistralai/Mistral-7B-v0.3"

# Train_DATASET_LIST=("base")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-llama3b-global-low-ppl")
# Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-llama3b-global-low-ppl-0.86")
Train_DATASET_LIST=("filtered-cured-50k-rho-baseline-llama3b-global-low-ppl" "filtered-cured-50k-rho-baseline-llama3b-global-high-ppl")
data_prop_list=(0.6)
# TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')

TASK_LISTS=('mbpp' "gsm8k")



for train_dataset_name in "${Train_DATASET_LIST[@]}" 
do
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
    model_tags=("${train_dataset_name}")
    ##############################################################


    if [[ "$train_dataset_name" == *"llama3b"* ]]; then
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
            ["squadv2"]="0 64 0.99"
            ["squad_completion"]="0 64 0.99"
            ["triviaqa"]="0 64 0.99"
            ["humaneval"]="0 64 0.99"
            ["mbpp"]="8 48 0.99"

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
            ["mbpp"]="8 48 0.99"

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
                ["mbpp"]="pretrained=${pretrained_model},dtype=bfloat16"

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
            # CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
            #     --data_dir raw_data/eval/tydiqa/ \
            #     --n_shot 1 \
            #     --max_num_examples_per_lang 100 \
            #     --max_context_length 512 \
            #     --save_dir $OUTPUT_PATH \
            #     --model_name_or_path $pretrained_model \
            #     --tokenizer_name_or_path $pretrained_model \
            #     --eval_batch_size 15 \
            #     --use_vllm

        done
    done
done 

echo "all experiments finished!!"

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
minutes=$((elapsed_time / 60))

echo "Elapsed time: $elapsed_time seconds"

