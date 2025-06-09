

## 1h
# bash run_reference_model.sh > zzz_reference_model.log 2>&1

# ## 4h
# bash run_full_baseline.sh > zzz_full_baseline.log 2>&1


## 4h
# bash fixed_model_cleaning.sh > zzz_fixed_model_cleaning.log 2>&1

## 4h
# bash run_rho_baseline_mistral.sh > run_rho_baseline_mistral_rerun.log 2>&1


## 5h
bash self_evolving_cleaning_mistral.sh > zzz_run-self_evolving_cleaning_mistral.log 2>&1


bash run_eval_token_selection_mistral.sh > zzz_run_eval_token_selection_mistral.log 2>&1

### rerun if time enough
# bash run_random_baseline.sh  > zzz_random_baseline.log 2>&1