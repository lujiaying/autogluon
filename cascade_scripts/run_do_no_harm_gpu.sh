# Jun 19
date="20220712"
dataset_name="PetFinder"
exp_name="ag-${date}_${dataset_name}_MM"
for infer_limit in 5e-3 2e-3 1e-3 5e-4 3e-4; do
python -m cascade_scripts.do_no_harm \
    --do_multimodal \
    --time_limit 3600 \
    --hpo_score_func_name ACCURACY --infer_time_limit ${infer_limit} \
    --dataset_name ${dataset_name} \
    --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
    --model_save_path AutogluonModels/${dataset_name}/${exp_name}
done
# --hpo_score_func_name ACCURACY --infer_time_limit 1e-3 \
# python -m cascade_scripts.convert_exp_result_to_latex \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --perf_metric_name accuracy


# for infer_limit in 5e-3 2e-3 1e-3 5e-4 3e-4; do
#     python -m cascade_scripts.ag_infer_limit \
#         --dataset_name ${dataset_name} \
#         --do_multimodal \
#         --time_limit 3600 \
#         --infer_time_limit ${infer_limit} \
#         --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#         --model_save_path AutogluonModels/${dataset_name}/${exp_name}
# done


# date="20220620"
# dataset_name="CPP-6aa99d1a"
# exp_name="ag-${date}_${dataset_name}_MM"
# python -m cascade_scripts.do_no_harm \
#     --do_multimodal \
#     --hpo_score_func_name ACCURACY --infer_time_limit 1e-3 \
#     --dataset_name ${dataset_name} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_MultiModal
# --hpo_score_func_name ACCURACY --infer_time_limit 4e-4 \

# python -m cascade_scripts.ag_infer_limit \
#      --do_multimodal \
#      --infer_time_limit 16e-4 \
#      --dataset_name ${dataset_name} \
#      --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#      --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_MultiModal

# python -m cascade_scripts.convert_exp_result_to_latex \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --perf_metric_name roc_auc
