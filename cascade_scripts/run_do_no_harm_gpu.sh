# Jun 19
# date="20220712"
# dataset_name="PetFinder"
# exp_name="ag-${date}_${dataset_name}_MM"
# for infer_limit in 5e-3 2e-3 1e-3 5e-4 3e-4; do
# python -m cascade_scripts.do_no_harm \
#     --do_multimodal \
#     --time_limit 3600 \
#     --hpo_score_func_name ACCURACY --infer_time_limit ${infer_limit} \
#     --dataset_name ${dataset_name} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}
# done
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


date="20220802"
presets="high_quality"
exp_name="ag-${date}_${dataset_name}_${presets}_1h_MM"
#dataset_name="CPP-6aa99d1a"
#for dataset_name in "CPP-6aa99d1a" "CPP-0e097514" "CPP-2d91e357" "CPP-e4367988" "CPP-a385488d"; do
for dataset_name in "CPP-a385488d"; do
    python -m cascade_scripts.do_no_harm \
        --do_multimodal \
        --predictor_presets ${presets} \
        --time_limit 3600 \
        --dataset_name ${dataset_name} \
        --exp_result_save_path ExpResults/${dataset_name}/${exp_name}_goodness_adjust_weights.csv \
        --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_${presets}_MultiModal
done
# --hpo_score_func_name ACCURACY --infer_time_limit 4e-4 \

# date="20220802"
# presets="high_quality"
# for infer_limit in 0.005 0.01 0.1; do
#     for dataset_name in "CPP-6aa99d1a" "CPP-0e097514" "CPP-2d91e357" "CPP-e4367988" "CPP-a385488d"; do
#         python -m cascade_scripts.do_no_harm \
#             --do_multimodal \
#             --predictor_presets ${presets} \
#             --ag_fit_infer_limit ${infer_limit} \
#             --time_limit 3600 \
#             --dataset_name ${dataset_name} \
#             --exp_result_save_path ExpResults/${dataset_name}/ag-${date}_${dataset_name}_${presets}_agFitInferLimit_${infer_limit}_1h_MM.csv \
#             --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_${presets}_agFitInferLimit_${infer_limit}_1h_MM
#     done
# done

# python -m cascade_scripts.ag_infer_limit \
#      --do_multimodal \
#      --infer_time_limit 16e-4 \
#      --dataset_name ${dataset_name} \
#      --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#      --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_MultiModal

# python -m cascade_scripts.convert_exp_result_to_latex \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --perf_metric_name roc_auc
