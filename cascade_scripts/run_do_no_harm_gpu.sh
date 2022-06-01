# May 26
date="20220526"
dataset_name="CPP-3564a7a7"
python -m cascade_scripts.do_no_harm \
    --do_multimodal \
    --dataset_name ${dataset_name} \
    --model_save_path AutogluonModels/${dataset_name}/ag-${date}_${dataset_name}_MultiModal
