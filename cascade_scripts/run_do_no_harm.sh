# May 13
# exp_name="ag-20220513_Inc_Meidium_AddKNN"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name Inc \
#     --exp_result_save_path ExpResults/Inc/${exp_name}.csv \
#     --model_save_path AutogluonModels/Inc/${exp_name}

# exp_name="ag-20220513_CoverTypeMulti_Meidium_AddKNN"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name CoverTypeMulti \
#     --exp_result_save_path ExpResults/CoverTypeMulti/${exp_name}.csv \
#     --model_save_path AutogluonModels/CoverTypeMulti/${exp_name}

# May 25
# date="20220525"
# dataset_name="CPP-3564a7a7"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_High_non-refit

# May 26
# date="20220526"
# dataset_name="CPP-3564a7a7"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_High_non-refit

# May 31
# exp_name="ag-20220531_Inc_High"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name Inc \
#     --exp_result_save_path ExpResults/Inc/${exp_name}.csv \
#     --model_save_path AutogluonModels/Inc/${exp_name}

# Jun 1
# dataset_name="CoverTypeMulti"
# exp_name="ag-20220531_${dataset_name}_High"
# python -m cascade_scripts.do_no_harm \
#     --dataset_name ${dataset_name} \
#     --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
#     --model_save_path AutogluonModels/${dataset_name}/${exp_name}

dataset_name="Inc"
exp_name="ag-20220531_Inc_Best"
python -m cascade_scripts.do_no_harm \
    --dataset_name ${dataset_name} \
    --exp_result_save_path ExpResults/${dataset_name}/${exp_name}.csv \
    --model_save_path AutogluonModels/${dataset_name}/${exp_name}
