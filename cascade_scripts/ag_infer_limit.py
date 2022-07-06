"""
Date: May 12, 2022
Author: Jiaying Lu
"""
import os
import argparse
import time

import tqdm
import pandas as pd
from autogluon.tabular import TabularPredictor, FeatureMetadata
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.core.utils.time import sample_df_for_time_func

from .do_no_harm import image_id_to_path_cpp, image_id_to_path_petfinder, append_approach_exp_result_to_df
from .cascade_utils import load_dataset, get_exp_df_meta_columns, MAIN_METRIC_COL, SPEED


def main(args):
    dataset_name = args.dataset_name
    model_save_path = args.model_save_path
    do_multimodal = args.do_multimodal
    force_training = args.force_training
    exp_result_save_path = args.exp_result_save_path
    presets = args.predictor_presets
    time_limit = args.time_limit
    infer_time_limit = args.infer_time_limit
    ndigits = 4

    assert infer_time_limit is not None
    model_save_path = f'{model_save_path}_inferLimit{infer_time_limit}'
    print(f'automatically change model_save_path={model_save_path}')

    # TODO: basically align with do_no_harm.py
    for fold_i, n_repeats, train_data, val_data, test_data, label, image_col, eval_metric, model_hyperparameters in load_dataset(dataset_name):
        # TODO: currently only trial on one fold
        if fold_i is not None and fold_i > 0:
            break

        fit_kwargs = dict(
            train_data=train_data,
            tuning_data=val_data,
            hyperparameters=model_hyperparameters,
            presets=presets,
            time_limit=time_limit,
            infer_limit=infer_time_limit,  # added
        )
        if do_multimodal:
            # currently support PetFinder or CPP
            # update several fit kwargs
            hyperparameters = get_hyperparameter_config('multimodal')
            if dataset_name == 'PetFinder':
                # Following tutorial, to use first image for each row
                train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
                test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
                train_data[image_col] = train_data[image_col].apply(image_id_to_path_petfinder)
                test_data[image_col] = test_data[image_col].apply(image_id_to_path_petfinder)
            elif dataset_name.startswith('CPP-'):
                train_data[image_col] = train_data[image_col].apply(image_id_to_path_cpp)
                test_data[image_col] = test_data[image_col].apply(image_id_to_path_cpp)
            else:
                raise ValueError(f'Currently NOT support do_multimodal with dataset_name={dataset_name}')
            feature_metadata = FeatureMetadata.from_df(train_data)
            feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
            fit_kwargs['hyperparameters'] = hyperparameters
            fit_kwargs['feature_metadata'] = feature_metadata
            fit_kwargs['presets'] = None

        if fold_i is not None:
            model_save_path = f'{model_save_path}/fold{fold_i}'
        if force_training is True or (not os.path.exists(model_save_path)):
            predictor = TabularPredictor(
                label=label,
                eval_metric=eval_metric,
                path=model_save_path,
            )
            predictor.fit(**fit_kwargs)
        else:
            predictor = TabularPredictor.load(model_save_path, require_version_match=False)
        leaderboard = predictor.leaderboard().set_index('model')
        persisted_models = predictor.persist_models('best')
        best_model = predictor.get_model_best()
        meta_cols = get_exp_df_meta_columns(predictor._learner.problem_type)
        if os.path.exists(exp_result_save_path):
            exp_result_df = pd.read_csv(exp_result_save_path, index_col='model')
        else:
            exp_result_df = pd.DataFrame(columns=meta_cols).set_index('model').dropna()

        n_repeats = 3
        infer_limit_batch_size = 10000   # TODO: change if not default
        test_data_sampled = sample_df_for_time_func(df=test_data, sample_size=infer_limit_batch_size)
        infer_times = []
        # infer_times is based on infer_limit_batch_size
        for _ in tqdm.tqdm(range(n_repeats)):
            ts = time.time()
            predictor.predict_proba(test_data_sampled, model=best_model)
            te = time.time()
            infer_times.append(te-ts)
        infer_times = sum(infer_times) / len(infer_times)
        print(f'DEBUG {infer_times=} for {infer_limit_batch_size=}')
        infer_times_simulated = infer_times / len(test_data_sampled) * len(test_data)
        print(f'DEBUG {infer_times_simulated=} for {len(test_data)=}')
        # test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
        model_name = f'AG_{infer_time_limit}'
        row = leaderboard.loc[best_model]
        time_val = row['pred_time_val']
        score_val = row['score_val']
        pred_proba = predictor.predict_proba(test_data, model=best_model)
        test_metrics = append_approach_exp_result_to_df(exp_result_df, model_name, predictor, infer_times_simulated, pred_proba, test_data, label, time_val, score_val)
        exp_result_df = exp_result_df.sort_values(by=[meta_cols[MAIN_METRIC_COL], SPEED], ascending=False)
        print(exp_result_df.round(ndigits).reset_index())
        exp_result_save_dir = os.path.dirname(exp_result_save_path)
        if not os.path.exists(exp_result_save_dir):
            os.makedirs(exp_result_save_dir)
        exp_result_df.to_csv(exp_result_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exp arguments to set")
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--exp_result_save_path', type=str, required=True)
    parser.add_argument('--force_training', action='store_true')
    parser.add_argument('--do_multimodal', action='store_true')
    parser.add_argument('--predictor_presets', type=str, default='medium_quality')
    parser.add_argument('--time_limit', type=float, default=None, help='Training time limit in seconds')
    parser.add_argument('--infer_time_limit', type=float, default=None,
            help='infer time limit in seconds per row.')
    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)

