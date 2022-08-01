# Jun 1
# presets="medium_quality"
# dataset_name="Inc"
# exp_name="ag-20220601_${dataset_name}_${presets}"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --predictor_presets ${presets} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}

# presets="best_quality"
# dataset_name="Inc"
# exp_name="ag-20220601_${dataset_name}_${presets}"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --predictor_presets ${presets} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}

# This is buggy because w/ intelex (patch_sklearn())
# exp_name="20220601_Inc_w_intelex"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name Inc \
#     --exp_result_save_path ExpResults/Debug/${exp_name}.csv \
#     --model_save_path AutogluonModels/Debug/${exp_name}

# presets="medium_quality"
# dataset_name="CoverTypeMulti"
# exp_name="ag-20220601_${dataset_name}_${presets}"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --predictor_presets ${presets} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}
 
# presets="best_quality"
# dataset_name="CoverTypeMulti"
# exp_name="ag-20220712_${dataset_name}_${presets}_timeLimit1h"
# python -m cascade_scripts.do_no_harm \
#     --time_limit 3600 \
#     --dataset_name ${dataset_name} \
#     --predictor_presets ${presets} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}

out_dir="Inc_medium_installed_intelex"
python -m cascade_scripts.do_no_harm \
    --dataset_name Inc \
    --exp_result_save_path ExpResults/${out_dir}.csv \
    --model_save_path AutogluonModels/${out_dir}