import os
import argparse
from functools import partial
import time
from datetime import datetime
from typing import Tuple, List
from dataclasses import asdict

import pandas as pd
from .benchmark_cpp import get_parquet_path, load_cpp_dataset, image_id_to_path
from autogluon.tabular import TabularPredictor
import autogluon.core.metrics as metrics
from autogluon.tabular.predictor.cascade_do_no_harm import F2SP_Preset, GreedyP_Preset, CascadeConfig
from autogluon.tabular.predictor.cascade_do_no_harm import get_all_predecessor_model_names
from autogluon.core.utils.time import sample_df_for_time_func


def exec_fit_cascade(predictor: TabularPredictor, test_data: pd.DataFrame, 
                     cascade_algo_list: List[str],
                     infer_limit_batch_size: int, infer_limit_list: List[float],
                     ) -> pd.DataFrame:
    def get_cascade_config_WE_details(predictor: TabularPredictor, cascad_config: CascadeConfig):
        model_predecessors_dict = {}
        for model in cascad_config.model:
            model_predecessors = get_all_predecessor_model_names(predictor, model)
            model_predecessors_dict[model] = list(model_predecessors)
        return model_predecessors_dict

    # Start function
    metrics_mapping = dict(
        acc=metrics.accuracy,
        balacc=metrics.balanced_accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )
    metrics_mapping_r = {v.name: k for k, v in metrics_mapping.items()}
    cascade_results = []
    test_data_sampled = sample_df_for_time_func(df=test_data, sample_size=infer_limit_batch_size, 
                                                max_sample_size=infer_limit_batch_size)
    for infer_limit in infer_limit_list:
        for cascade_algo_name in cascade_algo_list:
            preset = F2SP_Preset() if cascade_algo_name == 'F2S+' else GreedyP_Preset()
            fit_cascade_params = {
                'raw_data_for_infer_speed': test_data,
                'infer_limit': infer_limit,
                'infer_limit_batch_size': infer_limit_batch_size,
                'hyperparameter_cascade': asdict(preset),
                'max_memory': 0.5,
            }
            cascd_hyper_name = f'{cascade_algo_name}_{infer_limit}'    # used in result df to distinguish different trials
            cascade_config = predictor.fit_cascade(**fit_cascade_params)
            print(f'[DEBUG] ret cascade_config={cascade_config}')
            if cascade_config is None:
                cascade_results.append(
                    {
                    'cascade_hyper_name': cascd_hyper_name,
                    }
                )
            else:
                infer_time, pred_probas = predictor.do_infer_with_cascade_conf(cascade_config, test_data)
                test_metrics = predictor.evaluate_predictions(test_data[predictor.label], pred_probas, silent=True)
                test_metrics = {metrics_mapping_r[k]: v for k, v in test_metrics.items() if k in metrics_mapping_r}
                infer_time_genuine, _ = predictor.do_infer_with_cascade_conf(cascade_config, test_data_sampled)
                print(f'[DEBUG] infer_time={infer_time}, genuine_time={infer_time_genuine}, test_metrics={test_metrics}')
                cascade_m_predecessors_dict = get_cascade_config_WE_details(predictor, cascade_config)
                cascade_results.append(
                    {
                    'cascade_hyper_name': cascd_hyper_name,
                    'training_duration': cascade_config.fit_time,
                    'predict_duration': infer_time,
                    'predict_duration_genuine': infer_time_genuine,
                    'sec_per_row': infer_time / len(test_data),
                    'genuine_sec_per_row': infer_time_genuine / infer_limit_batch_size,
                    **test_metrics,
                    'infer_limit': cascade_config.infer_limit,
                    'infer_limit_batch_size': cascade_config.infer_limit_batch_size,
                    'cascade_config': asdict(cascade_config),
                    'weighted_ensemble_info': cascade_m_predecessors_dict,
                    }
                )
    return pd.DataFrame.from_records(cascade_results)


def main(args: argparse.Namespace):
    image_id_to_path_cpp = partial(image_id_to_path, args.cpp_img_dir, 'jpg')
    # traversal dir
    for cpp_session in os.listdir(args.cpp_result_dir):
        if args.session_names != None and cpp_session not in args.session_names:
            print(f'[DEBUG] skip {cpp_session}')
            continue
        # whether scores already exist
        cascade_result_path = os.path.join(args.cpp_result_dir, cpp_session, 'scores', args.cascade_result_fname)
        if os.path.exists(cascade_result_path):
            print(f'[DEBUG] skip {cpp_session} because already have cascade_result')
            continue
        test_fpath = get_parquet_path(os.path.join(args.cpp_dir, cpp_session, 'test')) 
        test_data, feature_metadata = load_cpp_dataset(test_fpath, image_id_to_path_cpp)
        predictor = TabularPredictor.load(os.path.join(args.cpp_result_dir, cpp_session, 'models'))
        cascade_results = exec_fit_cascade(predictor, test_data, args.cascade_algo_list, 
                args.infer_limit_batch_size, args.infer_limit_list)
        print(f'cascade_results={cascade_results}')
        cpp_ori_result_row = pd.read_csv(os.path.join(args.cpp_result_dir, cpp_session, 'scores/result.csv')).loc[0]
        eval_metric = cpp_ori_result_row['metric']
        cascade_results['id'] = [f'CPP{cpp_session}' for _ in range(len(cascade_results))]
        cascade_results['task'] = [cpp_session for _ in range(len(cascade_results))]
        cascade_results['framework'] = [cpp_ori_result_row['framework'] for _ in range(len(cascade_results))]
        cascade_results['constraint'] = [cpp_ori_result_row['constraint'] for _ in range(len(cascade_results))]
        cascade_results['fold'] = [-1 for _ in range(len(cascade_results))]
        cascade_results['type'] = ['binary' for _ in range(len(cascade_results))]
        cascade_results['metric'] = [eval_metric for _ in range(len(cascade_results))]
        cascade_results['score'] = [cascade_results[eval_metric][_] for _ in range(len(cascade_results))]
        cascade_results.to_csv(cascade_result_path, index=False)
        print(f'[INFO] cascade results written into {cascade_result_path}')


def none_or_float(value: str):
    if value == "None":
        return None
    else:
        return float(value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Execute fit_cascade() on saved artifacts")
    # required arguments
    parser.add_argument('--cpp_result_dir', type=str, required=True)
    # optional arguments
    parser.add_argument('--cpp_dir', type=str, default='datasets/cpp_research_corpora/2021_60datasets')
    parser.add_argument('--cpp_img_dir', type=str, default='/home/ec2-user/autogluon/datasets/cpp_research_corpora/2021_60datasets_imgs_raw')
    parser.add_argument('--cascade_result_fname', type=str, default='cascade_results.csv')
    parser.add_argument('--cascade_algo_list', nargs="+", type=str, default=['F2S+'])
    parser.add_argument('--infer_limit_batch_size', type=int, default=10000)
    parser.add_argument('--infer_limit_list', nargs='+', type=none_or_float, default=[None, 5e-3, 1e-3])
    parser.add_argument('--session_names', type=str, nargs='+', default=None)

    args = parser.parse_args()
    print(f'[INFO] Exp arguments {args}')
    main(args)