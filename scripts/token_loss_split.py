import torch
import os
import fire


def main(
        data_path = "results/loss/",
        root_path = "/mnt/data1/jinlong/token_selection_output",
        base_model_name_or_path="meta-llama/Llama-3.2-3B",
        train_dataset_name="test",
        loss_base_data_tag = "filtered-cured-50k-rho-baseline",
        num_subset = 5,
        ):
    

    if "3B" in base_model_name_or_path:
        base_model_tag = "llama3b"
    elif "8B" in base_model_name_or_path:
        base_model_tag = "llama8B"
    elif "7B" in base_model_name_or_path:
        base_model_tag = "mistral"
    else:
        print("unknown base model.")
    reference_model_name_or_path= root_path + f"{root_path}/models/{base_model_name_or_path}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup-{base_model_tag}"


    #### llama-8b
    # base_model_name_or_path="meta-llama/Llama-3.1-8B"
    # reference_model_name_or_path=f"/mnt/data1/jinlong/token_selection_output/models/{base_model_name_or_path}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup"

    #### mistral
    # base_model_name_or_path="mistralai/Mistral-7B-v0.3"
    # reference_model_name_or_path=f"/mnt/data1/jinlong/token_selection_output/models/{base_model_name_or_path}/data_prop_0.6/lora_merged_filtered-cured-10k-warmup"


    ### dataset name that needed
    # train_dataset_name="filtered-cured-50k-global-fixed-base-loss"
    # train_dataset_name="filtered-cured-50k-active-split-token_ranking_sample-fixed-base-loss"
    # train_dataset_name="filtered-cured-50k-active-split-global-half-positive-fixed-base-loss"
    # train_dataset_name="filtered-cured-50k-active-split-global-curve-positive-fixed-base-loss-using-warmup-full-data"
    # train_dataset_name="filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b"


    ref_name = os.path.basename(reference_model_name_or_path)
    base_name =  os.path.basename(base_model_name_or_path)

    reference_losses = torch.load(data_path + f"token_losses_{loss_base_data_tag}_{ref_name}.pt")
    base_losses = torch.load(data_path + f"token_losses_{loss_base_data_tag}_{base_name}.pt")

    subset_size = int(len(reference_losses) / num_subset)

    for idx in range(num_subset):
        
        train_dataset_name_tag = f"{train_dataset_name}_{idx}"
        
        subset_ref_losses = reference_losses[idx*subset_size: (idx+1)*subset_size]
        subset_base_losses = base_losses[idx*subset_size: (idx+1)*subset_size]
        
        merged_base_model_name = f"lora_merged_{train_dataset_name}_{idx-1}" if idx > 0 else base_name
        
        torch.save(subset_base_losses, data_path + f"token_losses_{train_dataset_name_tag}_{merged_base_model_name}.pt")
        torch.save(subset_ref_losses, data_path + f"token_losses_{train_dataset_name_tag}_{ref_name}.pt")


if __name__ == "__main__":
    fire.Fire(main)