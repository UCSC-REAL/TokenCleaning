import torch
import fire



def main(label_path='results/label/',
        num_subset=5,
        iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_llama8b",
        warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b",
        combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b",
        ):

    ## combination 1
    # iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop"
    # warmup_train_data_tag = "filtered-cured-50k-active-split-global-curve-positive-fixed-base-loss-using-warmup"
    # combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_combine_active-split-global-curve-positive-fixed-base-loss-using-warmup-label"

    ## combination 2
    # iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop"
    # warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss"
    # combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_combine_active-split-global-curve-positive-fixed-base-loss-label" ## wrong text label, which should global-half-positive

    # ## combination 4
    # iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3"
    # warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss"
    # combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-label"


    # ## llama-8b
    # iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_llama8b"
    # warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-llama8b"
    # combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_llama8b"


    ### mistral-7b
    # iter_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_mistral"
    # warmup_train_data_tag = "filtered-cured-50k-active-split-global-half-positive-fixed-base-loss-using-warmup-mistral"
    # combine_train_data_tag = "filtered-cured-50k-iter-split-global_data_prop_0.3_combine_active-split-global-half-positive-fixed-base-loss-using-warmup-label_mistral"



    for idx in range(1, num_subset):
        orig_labels_all =  torch.load(label_path + f"token_labels_{warmup_train_data_tag}_{idx}.pt")
        iter_labels_all =  torch.load(label_path + f"token_labels_{iter_train_data_tag}_{idx}.pt")

        selected_labels = [[-100 for _ in range(len(label))] for label in orig_labels_all]
        
        for i, (orig_labels_per_sample, iter_labels_per_sample) in enumerate(zip(orig_labels_all, iter_labels_all)):
            for j, (cur_label, additional_label) in enumerate(zip(orig_labels_per_sample, iter_labels_per_sample)):
                if cur_label != -100 or additional_label != -100:
                    selected_labels[i][j] = cur_label if cur_label != -100 else additional_label
            
        
        selected_labels_path = label_path + f"token_labels_{combine_train_data_tag}_{idx}.pt"
        torch.save(selected_labels, selected_labels_path)
        print(f"store the combine labels in file: {selected_labels_path}")

if __name__ == "__main__":
    fire.Fire(main)