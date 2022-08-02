"""
use to produce result.csv of automlbenchmark saved AG artifacts
"""

import os
import subprocess
from typing import Tuple
from collections import Counter
import argparse
from dataclasses import dataclass, asdict
import json
import time
import datetime

import pandas as pd
import openml
import tqdm
from autogluon.tabular import TabularPredictor
from autogluon.tabular import __version__ as ag_version
import autogluon.core.metrics as metrics
from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch
from autogluon.tabular.predictor.cascade_do_no_harm import INFER_UTIL_N_REPEATS, F2SP_Preset, GreedyP_Preset, CascadeConfig
from autogluon.tabular.predictor.cascade_do_no_harm import get_all_predecessor_model_names

#from .do_no_harm import fit_cascade, do_infer_with_cascade_conf, get_all_predecessor_model_names
#from .do_no_harm import INFER_UTIL_N_REPEATS, F2SP_Preset, GreedyP_Preset, CascadeConfig

@dataclass(frozen=True)
class ResultRow:
    id: str
    task: str
    framework: str
    constraint: str
    fold: int
    type: str
    result: float
    metric: str
    mode: str
    version: str
    params: str    # store cascade param, a jsonized dict
    app_version: str
    utc: str
    duration: float
    training_duration: float   # indicate fit_cascade duration
    predict_duration: float
    models_count: int          # copy from AG original model count
    seed: int
    info: str                  # store cascade config
    acc: float
    auc: float
    balacc: float
    logloss: float
    mae: float
    r2: float
    rmse: float
    model_ensemble_count: int   # copy from AG original ensemble model count
    predict_genuine_duration: float


def get_app_version() -> str:
    commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
    url = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url']).decode('ascii').strip()
    return json.dumps([url, branch, commit_id])
    

def get_benchmark_tasks(benchmark_id: int = 271) -> pd.DataFrame:
    # benchmark_id = 271    # https://www.openml.org/search?type=benchmark&id=271
    benchmark = openml.study.get_suite(benchmark_id)
    tasks_df = openml.tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    #binary_tasks_df = tasks_df[tasks_df.NumberOfClasses <= 2]
    #multiclass_tasks_df = tasks_df[tasks_df.NumberOfClasses > 2] 
    return tasks_df


def get_openml_test_data(task_id: int, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task = openml.tasks.get_task(task_id)
    train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold, sample=0,)
    X, y = task.get_X_and_y(dataset_format="dataframe")
    #X_train = X.iloc[train_indices]
    #y_train = y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]
    return X_test, y_test


def get_predict_genuine_duration(predictor: TabularPredictor, infer_limit_batch_size: int, 
                                 test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Genuine duration is calculated by resampling dataset size into **infer_limit_batch_size**
    """
    global INFER_UTIL_N_REPEATS
    n_repeats = INFER_UTIL_N_REPEATS
    time_per_row_df, _ = get_model_true_infer_speed_per_row_batch(data=test_data, predictor=predictor,
                                                                  batch_size=infer_limit_batch_size, repeats=n_repeats, 
                                                                  silent=True)
    # cal best model genuine predict duration (end to end + feature transform)
    best_model_time_per_row = time_per_row_df.loc[predictor.get_model_best()]
    predict_genuine_duration = best_model_time_per_row['pred_time_test_with_transform'] * len(test_data)
    return predict_genuine_duration


def get_cascade_config_WE_details(predictor: TabularPredictor, cascad_config: CascadeConfig):
    model_predecessors_dict = {}
    for model in cascad_config.model:
        model_predecessors = get_all_predecessor_model_names(predictor, model)
        model_predecessors_dict[model] = list(model_predecessors)
    return model_predecessors_dict


def main(benchmark_result_dir: str, cascade_result_out_dir: str):
    tasks_df = get_benchmark_tasks().set_index('name')
    #'Greedy+_default': asdict(GreedyP_Preset(hpo_score_func='eval_metric')),
    #'F2S+_default': asdict(F2SP_Preset()),
    hyperparameter_cascade = {
        'F2S+_default': asdict(F2SP_Preset()),
        }
    infer_limit_batch_size = 10000
    app_version = get_app_version()
    metrics_mapping = dict(
        acc=metrics.accuracy,
        balacc=metrics.balanced_accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        log_loss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )
    metrics_mapping_r = {v.name: k for k, v in metrics_mapping.items()}

    # traverse each EC2 dir
    model_best_counter = Counter()
    failed_folders = []
    cascade_result_out_path = os.path.join(cascade_result_out_dir, 'result.csv')
    if not os.path.exists(cascade_result_out_dir):
        os.makedirs(cascade_result_out_dir)
    fwrite = open(cascade_result_out_path, 'w')
    for ec2_dir_name in os.listdir(benchmark_result_dir):
        duration_ts = time.time()
        utc = datetime.datetime.utcnow().isoformat(sep='T', timespec='seconds')
        ec2_dir_path = os.path.join(benchmark_result_dir, ec2_dir_name)
        if os.path.isfile(ec2_dir_path):
            continue
        if ec2_dir_name in ['scores', 'logs']:
            continue
        # debug purpose
        #if ec2_dir_name != 'aws.ag.1h8c_gp3.car.2.agv053_jul30_high':
        if ec2_dir_name != 'aws.ag.1h8c_gp3.dionis.6.agv053_jul30_high':   # no WE_L2
            continue
        print(f'current is at {ec2_dir_name}')
        # debug Done
        result_path = os.path.join(ec2_dir_path, 'output', 'results.csv')
        result_df = pd.read_csv(result_path)
        model_dir_path = os.path.join(ec2_dir_path, 'output', 'models')
        if not os.path.exists(model_dir_path):
            failed_folders.append(model_dir_path)
            # directly use original result_df
            fwrite.write(result_df.to_csv(index=False))
            continue
        # get task name and fold
        temp_subdirs = os.listdir(model_dir_path)
        assert len(temp_subdirs) == 1
        task_name = temp_subdirs[0]
        if not task_name in tasks_df.index:
            continue
        task_dir_path = os.path.join(model_dir_path, task_name)
        temp_subdirs = os.listdir(task_dir_path)
        assert len(temp_subdirs) == 1
        fold = temp_subdirs[0]
        fold_dir_path = os.path.join(task_dir_path, fold)
        task_row = tasks_df.loc[task_name]
        task_id = task_row['tid'].item()
        temp_subdirs = os.listdir(fold_dir_path)
        if len(temp_subdirs) == 1:
            # means we need unzip models.zip
            assert temp_subdirs[0] == 'models.zip'
            zip_path = os.path.join(fold_dir_path, 'models.zip')
            #print(f'unzip: {zip_path} ...')
            subprocess.run(['unzip', '-q', zip_path, '-d', fold_dir_path])
        # print(task_row)
        predictor = TabularPredictor.load(fold_dir_path, require_version_match=False)
        best_model_name = predictor.get_model_best()
        model_best_counter[best_model_name] += 1
        fit_cascade_params = {
            'infer_limit': None,
            'infer_limit_batch_size': infer_limit_batch_size,
            'hyperparameter_cascade': hyperparameter_cascade,
        }
        train_duration_ts = time.time()
        cascade_configs_dict = predictor.fit_cascade(**fit_cascade_params)
        train_duration_te = time.time()
        test_X, test_y = get_openml_test_data(task_id, int(fold))
        for cascd_hyper_name, cascade_config in cascade_configs_dict.items():
            infer_time, pred_probas = predictor.do_infer_with_cascade_conf(cascade_config, test_X)
            test_metrics = predictor.evaluate_predictions(test_y, pred_probas, silent=True)
            predict_genuine_duration = get_predict_genuine_duration(predictor, infer_limit_batch_size, pd.concat([test_X, test_y], axis=1))
            duration_te = time.time()
            print(f'{cascd_hyper_name}, {cascade_config}, {infer_time}, {test_metrics}, {predict_genuine_duration}')
        # Start replace result_df
        result_df['params'] = json.dumps(fit_cascade_params)
        result_df['version'] = ag_version
        result_df['app_version'] = app_version
        result_df['utc'] = utc
        result_df['duration'] = duration_te - duration_ts
        result_df['training_duration'] = train_duration_te - train_duration_ts
        """
        result_df['predict_duration'] = infer_time
        result_df['predict_genuine_duration'] = predict_genuine_duration
        for metric_name, metric_val in test_metrics.items():
            if metric_name not in metrics_mapping_r:
                continue
            result_df[metrics_mapping_r[metric_name]] = metric_val
        cascade_model_predecessors_dict = get_cascade_config_WE_details(predictor, cascade_config)
        result_df['info'] = json.dumps({**asdict(cascade_config), **{'WE_predecessors_info': cascade_model_predecessors_dict}})
        """
        # End replace result_df
        fwrite.write(result_df.to_csv(index=False))
        exit(0)
    fwrite.close()
    print(f'model_best_counter={model_best_counter}')
    print(f'failed_folders={failed_folders}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_result_dir', type=str)
    parser.add_argument('cascade_result_dir', type=str)
    args = parser.parse_args()

    main(args.benchmark_result_dir, args.cascade_result_dir)