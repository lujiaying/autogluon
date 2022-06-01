# May 15
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name CAT \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_CAT
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name FASTAI \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_FASTAI
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name KNN \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_KNN


# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name NN_TORCH \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_NN_TORCH
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name GBM \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_GBM
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name CoverTypeMulti \
#     --load_from_ckpt \
#     --model_name XGB \
#     --model_save_path AutogluonModels/CoverTypeMulti/ag-20220515_CoverTypeMulti_baseline_XGB


# May 16

# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name CAT \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_CAT
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name FASTAI \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_FASTAI

# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name KNN \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_KNN
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name NN_TORCH \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_NN_TORCH

# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name GBM \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_GBM
# 
# python cascade_scripts/single_model_baselines.py \
#     --dataset_name Inc \
#     --load_from_ckpt \
#     --model_name XGB \
#     --model_save_path AutogluonModels/Inc/ag-20220516_Inc_baseline_XGB

# May 24
# for model_name in GBM CAT XGB RF XT KNN NN_TORCH FASTAI;
# for model_name in AG_TEXT_NN AG_IMAGE_NN;
# dataset_name="CPP-6aa99d1a"
# for model_name in AG_IMAGE_NN;
# do
#     python -m cascade_scripts.single_model_baselines \
#         --dataset_name ${dataset_name} \
#         --model_name ${model_name} \
#         --model_save_path AutogluonModels/${dataset_name}/ag-20220524_${dataset_name}_baseline_${model_name}
# done

# May 25
# for model_name in GBM CAT XGB RF XT KNN NN_TORCH FASTAI;
dataset_name="CPP-3564a7a7"
for model_name in KNN;
do
    python -m cascade_scripts.single_model_baselines \
        --dataset_name CPP-3564a7a7 \
        --model_name ${model_name} \
        --model_save_path AutogluonModels/${dataset_name}/ag-20220525_${dataset_name}_baseline_${model_name}
done
