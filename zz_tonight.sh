########################################################################
### 6h ##

# echo "LlaMA-3.1-3B:: zzz_llama3b-combine-case full case combine"
# bash run_iter_pattern.sh > zzz_llama3b-combine-case.log 2>&1

# ########################################################################

# echo "LlaMA-3.1-3B:: four case evaluation"
# bash run_eval_token_selection.sh > zzz_llama3b-combine-case-eval 2>&1

########################################################################
# echo "LlaMA-3.1-3B:: zzz_llama3b-warmup0.6_combine_iter0.6"
# bash run_ours_pipeline.sh > zzz_llama3b-warmup0.6_combine_iter0.3.log 2>&1

# ########################################################################

# echo "LlaMA-3.1-3B:: zzz_llama3b-warmup0.6_combine_iter0.6 eval"
# bash run_eval_token_selection.sh > zzz_llama3b-warmup0.6_combine_iter0.3-eval.log 2>&1

# ########################################################################
# # 1.5 h
# echo "LlaMA-3.1-3B:: zzz_llama3b-warmup0.45_combine_iter0.3.log"
# bash run_ours_pipeline.sh > zzz_llama3b-warmup0.45_combine_iter0.3.log 2>&1

# #######################################################################
# # 2 h
# echo "LlaMA-3.1-8B:: zzz_llama8b-run_rho_baseline-with-prompt.log"
# bash run_rho_baseline.sh > zzz_llama8b-run_rho_baseline-with-prompt.log 2>&1

# ########################################################################
# # 2h
# echo "Mistral:: zzz_mistral7b-run_rho_baseline-with-prompt.log"
# bash run_rho_baseline-mistral.sh > zzz_mistral7b-run_rho_baseline-with-prompt.log 2>&1

# ########################################################################
# # 1h
# echo "LlaMA-3.1-3B eval:: zz_llama3b-warmup0.45_combine_iter0.3-eval.log"
# bash run_eval_token_selection.sh > zzz_llama3b-warmup0.45_combine_iter0.3-eval.log 2>&1

# #########################################################################
# # 1h
# echo "rho baseline eval (mistral and llama8b):: zzz_llama8b_mistral_rho_baseline_with_prompt-eval.log"
# bash run_eval_token_selection_temp.sh > zzz_llama8b_mistral_rho_baseline_with_prompt-eval.log 2>&1

# #########################################################################

# 1.5 h
# echo "LlaMA-3.1-3B:: zzz_llama3b-iter0.6_combine_warmup0.3.log"

# bash run_ours_pipeline.sh > zzz_llama3b-iter0.6_combine_warmup0.3.log 2>&1
# # 1.5 h

# bash run_ours_pipeline_temp.sh > zzz_llama3b-iter0.6_combine_warmup0.6.log 2>&1

# # #########################################################################
# # 2h
# echo "LlaMA-3.1-3B eval:: zzz_llama3b-iter0.6_combine_warmup0.6_0.3_eval.log"
# bash run_eval_token_selection.sh > zzz_llama3b-iter0.6_combine_warmup0.6_0.3_eval.log 2>&1

# #########################################################################
# sleep 30m

# bash run_iter_pattern_new.sh > zzz_llama3b_iter0.8_new_fixed-base-model.log 2>&1

# bash run_iter_pattern_new_1.sh > zzz_llama3b_iter0.7_new_fixed-base-model.log 2>&1


# bash run_ours_pipeline.sh > zzz_llama3b-iter0.6-fixed-base-model_combine_warmup-global-half-pos.log 2>&1

# bash run_eval_token_selection_1.sh > zzz_llama3b_iter_diff_prop_new_fixed_base_model.log 2>&1

# bash run_eval_token_selection_2.sh > zzz_llama3b-iter0.6-fixed-base-model_combine_warmup-global-half-pos-eval.log 2>&1


# #########################################################################


# ### two models 5h
# bash run_iter_pattern_new.sh > zzz_llama8b_mistral_iter_split-new.log 2>&1

# # 2.5h
# bash run_rho_baseline_global_llama8b.sh > zzz_rho_baseline_global_llama8b.log 2>&1

# # 2.5h
# bash run_rho_baseline_global_mistral.sh > zzz_rho_baseline_global_mistral.log 2>&1

# ##eval
# # 20m
# bash run_eval_token_selection_rho_baseline_global.sh > zzz_rho_baseline_global_llama8b_mistral-eval.log 2>&1

# # 1.5h
# bash run_eval_token_selection.sh > zzz_llama8b_mistral_iter_split-new-eval.log 2>&1

# # 3h
# bash run_iter_pattern_new_1.sh > zzz_explore_data_prop.log 2>&1

# # 30m
# bash run_eval_token_selection_1.sh > zzz_explore_data_prop-eval.log 2>&1


# #########################################################################
# #########################################################################

# bash run_iter_pattern_new_300k.sh > zzz_iter_new_300k.log 2>&1

# bash run_eval_token_selection.sh > zzz_eval_iter_new_300k.log 2>&1

# #########################################################################
# #########################################################################


# bash run_iter_pattern_new_continue.sh > zzz_iter_continue_training_50k.log 2>&1

# bash run_eval_token_selection.sh > zzz_eval_iter_new_continue_training_50k.log 2>&1


bash run_rho_baseline_sample.sh > zzz_rho-diff_data_prop.log 2>&1

bash run_eval_token_selection_rho_baseline_global.sh > zzz_rho-diff_data_prop-eval.log 2>&1