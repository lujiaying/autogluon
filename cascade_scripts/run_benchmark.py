"""
use to produce result.csv of automlbenchmark saved AG artifacts
"""

import os
import subprocess
from typing import Tuple

import pandas as pd
import openml
from autogluon.tabular import TabularPredictor

from .do_no_harm import fit_cascade

def get_benchmark_tasks(benchmark_id: int = 271) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # benchmark_id = 271    # https://www.openml.org/search?type=benchmark&id=271
    benchmark = openml.study.get_suite(benchmark_id)
    
    tasks_df = openml.tasks.list_tasks(output_format="dataframe", task_id=benchmark.tasks)
    binary_tasks_df = tasks_df[tasks_df.NumberOfClasses <= 2]
    multiclass_tasks_df = tasks_df[tasks_df.NumberOfClasses > 2] 
    return binary_tasks_df, multiclass_tasks_df


if __name__ == '__main__':
    result_root_dir_path = '/home/ec2-user/automlbenchmark/results/autogluon_v0_5_1_high_saveall.openml_bench_271-binary.1h8c_gp3.aws.20220728T050522'
    binary_task_df, multiclass_task_df = get_benchmark_tasks()
    print('binary_task_df:')
    print(binary_task_df)
    binary_task_df = binary_task_df.set_index('name')

    # traverse each EC2 dir
    for ec2_dir_name in os.listdir(result_root_dir_path):
        ec2_dir_path = os.path.join(result_root_dir_path, ec2_dir_name)
        if os.path.isfile(ec2_dir_path):
            continue
        model_dir_path = os.path.join(ec2_dir_path, 'output', 'models')
        # get task name and fold
        temp_subdirs = os.listdir(model_dir_path)
        assert len(temp_subdirs) == 1
        task_name = temp_subdirs[0]
        task_dir_path = os.path.join(model_dir_path, task_name)
        temp_subdirs = os.listdir(task_dir_path)
        assert len(temp_subdirs) == 1
        fold = temp_subdirs[0]
        fold_dir_path = os.path.join(task_dir_path, fold)
        task_row = binary_task_df.loc[task_name]
        temp_subdirs = os.listdir(fold_dir_path)
        if len(temp_subdirs) == 1:
            # means we need unzip models.zip
            assert temp_subdirs[0] == 'models.zip'
            zip_path = os.path.join(fold_dir_path, 'models.zip')
            print(f'unzip: {zip_path} ...')
            subprocess.run(['unzip -q', zip_path, '-d', fold_dir_path])
        # print(task_row)
        # TODO: how to constrain to 1h8c 8 cores?
        predictor = TabularPredictor.load(fold_dir_path, require_version_match=False)
        fit_cascade(predictor)
        exit(0)