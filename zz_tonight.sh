

# #########################################################################
# #########################################################################
##### mistral #####
### 2.5h for each
bash run_full_baseline_mistral_41.sh > zzz_run_full_baseline_mistral_41.log 2>&1
bash run_full_baseline_mistral_43.sh > zzz_run_full_baseline_mistral_43.log 2>&1

# bash run_random_baseline_mistral_41.sh > zzz-run_random_baseline_mistral_41.log 2>&1
# bash run_random_baseline_mistral_43.sh > zzz-run_random_baseline_mistral_43.log 2>&1

bash run_rho_baseline_sample_mistral_41.sh > zzz-run_rho_baseline_sample_mistral_41.log 2>&1
bash run_rho_baseline_sample_mistral_43.sh > zzz-run_rho_baseline_sample_mistral_43.log 2>&1

bash run_rho_baseline_global_mistral_41.sh > zzz-run_rho_baseline_global_mistral_41.log 2>&1
bash run_rho_baseline_global_mistral_43.sh > zzz-run_rho_baseline_global_mistral_43.log 2>&1

# ##0.5h
bash run_ds2_reference_model_mistral_41.sh > zzz-run_ds2_reference_model_mistral_41.log 2>&1
bash run_ds2_reference_model_mistral_43.sh > zzz-run_ds2_reference_model_mistral_43.log 2>&1


# bash run_iter_pattern_new_mistral_41.sh > zzz-run_iter_pattern_new_mistral_41.log 2>&1
# bash run_iter_pattern_new_mistral_43.sh > zzz-run_iter_pattern_new_mistral_43.log 2>&1

# #########################################################################
# #########################################################################
##### llama8b #####
### 2h for each
# bash run_full_baseline_llama8b_41.sh > zzz_run_full_baseline_llama8b_41.log 2>&1
# bash run_full_baseline_llama8b_43.sh > zzz_run_full_baseline_llama8b_43.log 2>&1


# bash run_random_baseline_llama8b_41.sh > zzz-run_random_baseline_llama8b_41.log 2>&1
# bash run_random_baseline_llama8b_43.sh > zzz-run_random_baseline_llama8b_43.log 2>&1


# bash run_rho_baseline_sample_llama8b_41.sh > zzz-run_rho_baseline_sample_llama8b_41.log 2>&1
# bash run_rho_baseline_sample_llama8b_43.sh > zzz-run_rho_baseline_sample_llama8b_43.log 2>&1


# bash run_ds2_reference_model_llama8b_41.sh > zzz-run_ds2_reference_model_llama8b_41.log 2>&1
# bash run_ds2_reference_model_llama8b_43.sh > zzz-run_ds2_reference_model_llama8b_43.log 2>&1

# bash run_rho_baseline_global_llama8b_43.sh > zzz-run_rho_baseline_global_llama8b_43.log 2>&1
# bash run_rho_baseline_global_llama8b_41.sh > zzz-run_rho_baseline_global_llama8b_41.log 2>&1


# bash run_iter_pattern_new_llama8b_41.sh > zzz-run_iter_pattern_new_llama8b_41.log 2>&1
# bash run_iter_pattern_new_llama8b_43.sh > zzz-run_iter_pattern_new_llama8b_43.log 2>&1

# bash run_eval_token_selection_rho_baseline_global.sh > zzz-mistral_eval_continue.log 2>&1

# bash run_eval_token_selection_llama8b.sh > zzz-llama8b_eval.log 2>&1


# #########################################################################
# #########################################################################
##### llama13b #####
### 3.5h for each


# bash run_ds2_reference_model_llama13b.sh > zzz-run_ds2_reference_model_llama13b.log 2>&1

# bash run_full_baseline_llama13b.sh > zzz-run_full_baseline_llama13b.log 2>&1


# bash run_random_baseline_llama13b.sh > zzz-run_random_baseline_llama13b.log 2>&1


# bash run_rho_baseline_global_llama13b.sh > zzz-run_rho_baseline_global_llama13b.log 2>&1


# bash run_rho_baseline_sample_llama13b.sh > zzz-run_rho_baseline_sample_llama13b.log 2>&1

# bash run_iter_pattern_new_llama13b.sh > zzz-run_iter_pattern_new_llama13b.log 2>&1

# bash run_eval_token_selection_llama13b.sh > zzz-run_eval_token_selection_llama13b.log 2>&1