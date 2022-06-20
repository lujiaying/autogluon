# May 26
# date="20220526"
# dataset_name="CPP-3564a7a7"
# python -m cascade_scripts.do_no_harm \
#     --do_multimodal \
#     --dataset_name ${dataset_name} \
#     --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_MultiModal

# Jun 19
date="20220619"
dataset_name="PetFinder"
exp_name="ag-${date}_${dataset_name}_MM"
python -m cascade_scripts.do_no_harm \
    --do_multimodal \
    --dataset_name ${dataset_name} \
    --hpo_score_func_name ACCURACY --infer_time_limit 1e-2 \
    --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
    --model_save_path AutogluonModels/${dataset_name}/${exp_name}
# python -m cascade_scripts.convert_exp_result_to_latex \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --perf_metric_name accuracy
