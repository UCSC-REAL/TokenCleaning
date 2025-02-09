import json
from datasets import load_dataset
import fire


def main(data_path = 'selected_data/',
        subset_size = 5,
        generate_train_data_name = "filtered-cured-50k-test",
        dataset_name = 'filtered-cured-50k',
        ):


    # dataset_name = 'filtered-cured-50k-shuffle'
    # dataset_name = 'filtered-cured-50k'
    # dataset_name = "random_subset_50k"
    # dataset_name = "alpaca_52k"
    # dataset_name = "full"
    # dataset_name = "filtered-cured-10k"
    
    if "filtered-cured-50k" in generate_train_data_name:
        dataset = load_dataset("json", data_files= data_path + f'{dataset_name}_dataset.json')['train']
    elif "full-300k" in generate_train_data_name:
        dataset = load_dataset("json", data_files= data_path + f'full-300k_dataset.json')['train']
    else:
        raise NotImplementedError

    # dataset = dataset.shuffle(seed=42) 
    # dataset.to_json(data_path + "filtered-cured-50k-shuffle.json")
    # dataset = dataset.select(range(len(dataset)-1, -1, -1))


    data_size = len(dataset) // subset_size
    
    for i in range(subset_size):
        selected_indices = [idx for idx in range(data_size *i, data_size * (i+1))]
        subset = dataset.select(selected_indices)
        
        subset.to_json(data_path + f"{generate_train_data_name}_{i}.json")



if __name__ == "__main__":
    fire.Fire(main)