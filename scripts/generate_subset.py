import json
from datasets import load_dataset
import fire
import os

def main(data_path = 'selected_data/',
        subset_size = 5,
        generate_train_data_name = "ds2-50k-test",
        dataset_name = 'ds2-50k',
        ):

    try:
        dataset = load_dataset(dataset_name)['train']
        print(f"Loaded dataset '{dataset_name}' from Hugging Face hub.")
    except Exception as e:
        print(f"Failed to load '{dataset_name}' from Hugging Face hub. Trying local path...")
        local_data_file = os.path.join(data_path, f"{dataset_name}.json")
        if os.path.exists(local_data_file):
            dataset = load_dataset('json', data_files=local_data_file)['train']
            print(f"Loaded dataset from local path: {local_data_file}")
        else:
            raise FileNotFoundError(f"Dataset not found on Hugging Face or locally: {dataset_name}")


    data_size = len(dataset) // subset_size
    
    for i in range(subset_size):
        selected_indices = [idx for idx in range(data_size *i, data_size * (i+1))]
        subset = dataset.select(selected_indices)
        
        subset.to_json(data_path + f"{generate_train_data_name}_{i}.json")


if __name__ == "__main__":
    fire.Fire(main)