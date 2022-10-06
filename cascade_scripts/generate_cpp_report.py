import os
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularDataset

from .benchmark_cpp import get_parquet_path


def gather_all_ag_results(exp_result_dir: str, out_path: str):
    cnt = 0
    all_dfs = []
    for cpp_session in os.listdir(exp_result_dir):
        result_path = os.path.join(exp_result_dir, cpp_session, 'scores/result.csv')
        df = pd.read_csv(result_path)
        all_dfs.append(df)
        cnt += 1
    print(f'get {cnt} cpp sessions')
    merged_df = pd.concat(all_dfs)
    merged_df.to_csv(out_path, index=False)


def gather_all_cascade_results(exp_result_dir: str, output_dir: str):
    in_dir_exp_name = exp_result_dir.split('/')[1]
    cnt = 0
    total_cnt = 0
    all_dfs = []
    for cpp_session in os.listdir(exp_result_dir):
        result_path = os.path.join(exp_result_dir, cpp_session, 'scores/cascade_results.csv')
        total_cnt += 1
        if not os.path.exists(result_path):
            continue
        df = pd.read_csv(result_path)
        all_dfs.append(df)
        cnt += 1
    print(f'get {cnt} cpp sessions contains cascade results')
    merged_df = pd.concat(all_dfs)
    for i, group in merged_df.groupby("cascade_hyper_name"):
        out_path = os.path.join(output_dir, f'{i}.{in_dir_exp_name}.csv')
        group.to_csv(out_path, index=False)
    print(f'[INFO] cascade results status: {cnt}/{total_cnt}')
    

def get_test_set_size(cpp_data_dir: str = 'datasets/cpp_research_corpora/2021_60datasets') -> Dict[str, int]:
    session_size_dict = {}
    for cpp_session in os.listdir(cpp_data_dir):
        test_fpath = get_parquet_path(os.path.join(cpp_data_dir, cpp_session, 'test'))
        dataset = TabularDataset(test_fpath)
        session_size_dict[cpp_session] = len(dataset)
    return session_size_dict


def generate_report():
    # ad-hoc input files, change manually
    in_dir = 'cpp_report'
    ag_high_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022-all_results.csv'))
    if 'predict_genuine_duration' in ag_high_df:
        ag_high_df['predict_duration_genuine'] = ag_high_df['predict_genuine_duration']
    cascd_goodness_df = pd.read_csv(os.path.join(in_dir, 'F2S+_None.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_5ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.005.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_1ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.001.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_goodness_df['framework'] = cascd_goodness_df['cascade_hyper_name']
    cascd_5ms_df['framework'] = cascd_5ms_df['cascade_hyper_name']
    cascd_1ms_df['framework'] = cascd_1ms_df['cascade_hyper_name']

    # filter to only show cascade exist sessions
    print(f'before ag_high_df len={ag_high_df.shape}')
    ag_high_df = ag_high_df[ag_high_df.task.isin(cascd_goodness_df.task)]
    print(f'after ag_high_df len={ag_high_df.shape}')

    # 1ms contains too much nan
    # all_df = pd.concat([ag_high_df, cascd_goodness_df, cascd_5ms_df, cascd_1ms_df])
    all_df = pd.concat([ag_high_df, cascd_goodness_df, cascd_5ms_df])
    # fill nan
    all_df['auc'] = all_df['auc'].fillna(0.5)
    # get sec per row
    session_size_dict = get_test_set_size()
    all_df['sec_per_row'] = all_df.apply(lambda row: row.loc['predict_duration'] / session_size_dict[row.loc['task']], axis=1)
    all_df['genuine_sec_per_row'] = all_df.apply(lambda row: row.loc['predict_duration_genuine'] / session_size_dict[row.loc['task']], axis=1)
    # get task wise mean results
    agg_df = all_df.groupby('framework').agg(
        mean_auc=pd.NamedAgg(column='auc', aggfunc=np.mean),
        mean_sec_per_row=pd.NamedAgg(column='sec_per_row', aggfunc=np.mean),
        mean_genuine_sec_per_row=pd.NamedAgg(column='genuine_sec_per_row', aggfunc=np.mean),
    )
    print(agg_df)
    
    # draw
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xscale('log')
    ax.set_title('CPP(4 hour, 8 cores)')
    sns.scatterplot(data=agg_df, x='mean_sec_per_row', y='mean_auc', hue='framework', s=100)
    ax.legend(loc='center left')
    fig.savefig('cpp_report/Sep30.png')
    plt.close(fig)


if __name__ == '__main__':
    exp_result_dir = 'ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022'
    out_dir = 'cpp_report'
    # gather_all_ag_results(exp_result_dir, os.path.join(out_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022-all_results.csv'))

    gather_all_cascade_results(exp_result_dir, out_dir)
    # generate_report()