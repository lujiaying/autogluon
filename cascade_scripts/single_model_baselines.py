"""
Date: May 15, 2022
Author: Jiaying Lu
"""
import os
import time
import argparse

from sklearnex import patch_sklearn
patch_sklearn()
import torch
from autogluon.tabular import TabularPredictor, TabularDataset, FeatureMetadata

from .do_no_harm import image_id_to_path


def main(args: argparse.Namespace):
    dataset_name = args.dataset_name
    model_save_path = args.model_save_path
    force_training = args.force_training
    model_name = args.model_name

    # Cover Type MultiClass
    if dataset_name == 'CoverTypeMulti':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification/'
        label = 'Cover_Type'
        path_train = path_prefix + 'train_data.csv'
        path_test = path_prefix + 'test_data.csv'
        eval_metric = 'accuracy'
    # Adult Income Dataset
    elif dataset_name == 'Inc':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
        label = 'class'
        path_train = path_prefix + 'train.csv'
        path_test = path_prefix + 'test.csv'
        eval_metric = 'roc_auc'
    elif dataset_name == 'CPP-6aa99d1a':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/6aa99d1a-1d4b-4d30-bd8b-a26f259b6482/'
        label = 'label'
        path_train = path_prefix + 'train/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        eval_metric = 'roc_auc'
    elif dataset_name == 'CPP-3564a7a7':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/3564a7a7-0e7c-470f-8f9e-5a029be8e616/'
        label = 'label'
        path_train = path_prefix + 'train/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        eval_metric = 'roc_auc'
    else:
        print(f'currently not support dataset_name={dataset_name}')
        exit(-1)

    train_data = TabularDataset(path_train)
    test_data = TabularDataset(path_test)

    fit_kwargs = dict(
        train_data=train_data,
        hyperparameters={
            model_name: {},
        },
    )
    # prepare image feature
    if model_name in ['AG_IMAGE_NN']:
        image_col = 'image_id'
        train_data[image_col] = train_data[image_col].apply(image_id_to_path)
        test_data[image_col] = test_data[image_col].apply(image_id_to_path)
        feature_metadata = FeatureMetadata.from_df(train_data)
        feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
        fit_kwargs['feature_metadata'] = feature_metadata
    
    if force_training is True or (not os.path.exists(model_save_path)):
        predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric,
            path=model_save_path,
        )
        predictor.fit(**fit_kwargs)
    else:
        predictor = TabularPredictor.load(model_save_path, require_version_match=False)

    predictor.persist_models('all')
    model_names = predictor.get_model_names()
    model_name_full = model_names[0]
    if model_name in ['AG_TEXT_NN']:
        # For CPU inference purpose
        model_obj = predictor._trainer.models[model_name_full]
        prev_num_gpus = model_obj.model._predictor._config.env.num_gpus
        print(f'{prev_num_gpus=}')
        if torch.cuda.device_count() < prev_num_gpus:
            model_obj.model._predictor._config.env.num_gpus = torch.cuda.device_count()
            print(f'reset to num_gpus={torch.cuda.device_count()}')
    # leaderboard = predictor.leaderboard(test_data)
    # model_name_full = leaderboard['model'].tolist()[0]

    # Normal approach
    ts = time.time()
    pred_proba = predictor.predict_proba(test_data, model=model_name_full)
    te = time.time()
    print('--------')
    print(f'Single Model: {model_name_full}')
    print(predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True))
    print(f'{te - ts}s')
    print('--------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exp arguments to set")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--force_training', action='store_true')
    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)
