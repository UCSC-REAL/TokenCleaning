export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUs=8



base_model="meta-llama/Llama-3.2-3B" #"meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3"
data_prop=0.6


## path 
result_path="eval_results"
cluster_root_path="/mnt/data1/jinlong/token_selection_output"
model_path=$cluster_root_path/$(basename "$base_model")/data_prop_${data_prop}



#### num_fewshot, batch_size, max_examples(less 1 means proportion)
declare -A TASK_PARAMS=(
    ["mmlu"]="5 16 0.99"
    ["truthfulqa"]="0 128 0.99"
    ["hellaswag"]="0 128 0.99"
    ["logiqa"]="0 32 0.99"
    ["boolq"]="0 32 0.99"
)

TASK_LISTS=('mmlu' "truthfulqa" "hellaswag" "arc_challenge" "boolq" 'logiqa')


### eval models
Train_DATASET_LIST=("ds2-50k-self-evolving") 




for train_dataset_tag in "${Train_DATASET_LIST[@]}";do

    if [[ $train_dataset_tag == 'base' ]]; then
        pretrained_model=$base_model
    else
        pretrained_model=${model_path}/lora_merged_${train_dataset_tag}
    fi

    OUTPUT_PATH=${result_path}/$(basename "$base_model")/${data_prop}/${train_dataset_tag}
    mkdir -p $OUTPUT_PATH

    for idx in "${!TASK_LISTS[@]}"; do

        task=${TASK_LISTS[$idx]}
        params=(${TASK_PARAMS[$task]})  
        num_fewshot=${params[0]}
        batch_size=${params[1]}
        max_examples_per_task=${params[2]}
        gpu_idx=$((idx % 8))
        model_args="pretrained=${pretrained_model},dtype=bfloat16"

        echo "Running task $task with num_fewshot=$num_fewshot, batch_size=$batch_size, max_examples per task= $max_examples_per_task"

        accelerate launch --multi-gpu --main_process_port 29519 --num_processes $NUM_GPUs \
                -m lm_eval --model hf \
                --model_args $model_args \
                --tasks $task \
                --batch_size $batch_size \
                --num_fewshot $num_fewshot \
                --limit $max_examples_per_task \
                --output_path $OUTPUT_PATH \
                --seed 42 \
                --trust_remote_code
                
    done

    ###########################################
    ############## tydiqa eval ################
    ###########################################
    CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
        --data_dir eval_data/eval/tydiqa/ \
        --n_shot 1 \
        --max_num_examples_per_lang 100 \
        --max_context_length 512 \
        --save_dir $OUTPUT_PATH \
        --model_name_or_path $pretrained_model \
        --tokenizer_name_or_path $pretrained_model \
        --eval_batch_size 5

done 



