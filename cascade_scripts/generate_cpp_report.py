import os
from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def gather_all_cascade_results(exp_result_dir: str, output_dir: str, cascd_result_rel_path: str = 'scores/cascade_results.csv'):
    in_dir_exp_name = exp_result_dir.split('/')[1]
    cnt = 0
    total_cnt = 0
    all_dfs = []
    for cpp_session in os.listdir(exp_result_dir):
        result_path = os.path.join(exp_result_dir, cpp_session, cascd_result_rel_path)
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
    import seaborn as sns
    # ad-hoc input files, change manually
    in_dir = 'cpp_report'
    ag_tab_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Oct112022-tabular_results.csv'))
    ag_high_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022-all_results.csv'))
    ag_il3ms_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-il3ms_results.csv'))
    ag_il5ms_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-il5ms_results.csv'))
    ag_il7ms_df = pd.read_csv(os.path.join(in_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep282022-il7ms_results.csv'))
    cascd_goodness_df = pd.read_csv(os.path.join(in_dir, 'F2S+_None.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_10ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.01.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_7ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.007.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_5ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.005.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_3ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.003.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    cascd_1ms_df = pd.read_csv(os.path.join(in_dir, 'F2S+_0.001.CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022.csv'))
    # rename framework
    cascd_goodness_df['framework'] = cascd_goodness_df['cascade_hyper_name']
    cascd_10ms_df['framework'] = cascd_10ms_df['cascade_hyper_name']
    cascd_7ms_df['framework'] = cascd_7ms_df['cascade_hyper_name']
    cascd_5ms_df['framework'] = cascd_5ms_df['cascade_hyper_name']
    cascd_3ms_df['framework'] = cascd_3ms_df['cascade_hyper_name']
    cascd_1ms_df['framework'] = cascd_1ms_df['cascade_hyper_name']

    # filter to only show cascade exist sessions
    # print(f'before ag_high_df len={ag_high_df.shape}')
    # ag_high_df = ag_high_df[ag_high_df.task.isin(cascd_goodness_df.task)]
    # print(f'after ag_high_df len={ag_high_df.shape}')

    # 1ms contains too much nan
    # all_df = pd.concat([ag_high_df, cascd_goodness_df, cascd_5ms_df, cascd_1ms_df])
    all_df = pd.concat([ag_tab_df, ag_high_df, ag_il3ms_df, ag_il5ms_df, ag_il7ms_df, 
                        cascd_goodness_df, cascd_3ms_df, cascd_5ms_df, cascd_7ms_df, cascd_10ms_df])
    # fill nan
    all_df['auc'] = all_df['auc'].fillna(0.5)
    # get sec per row
    session_size_dict = get_test_set_size()
    all_df['sec_per_row'] = all_df.apply(lambda row: row.loc['predict_duration'] / session_size_dict[row.loc['task']], axis=1)
    all_df['genuine_sec_per_row'] = all_df.apply(lambda row: row.loc['predict_duration_genuine'] / 10000, axis=1)
    # get task wise mean results
    agg_df = all_df.groupby('framework').agg(
        mean_auc=pd.NamedAgg(column='auc', aggfunc=np.mean),
        mean_sec_per_row=pd.NamedAgg(column='sec_per_row', aggfunc=np.mean),
        mean_genuine_sec_per_row=pd.NamedAgg(column='genuine_sec_per_row', aggfunc=np.mean),
    )
    print(agg_df)
    """
    # temp generate one session result
    temp_df = all_df[all_df.task == "3564a7a7-0e7c-470f-8f9e-5a029be8e616"]
    agg_df = temp_df.groupby('framework').agg(
        mean_auc=pd.NamedAgg(column='auc', aggfunc=np.mean),
        mean_sec_per_row=pd.NamedAgg(column='sec_per_row', aggfunc=np.mean),
        mean_genuine_sec_per_row=pd.NamedAgg(column='genuine_sec_per_row', aggfunc=np.mean),
    )
    print(agg_df.to_records())
    exit(0)
    """
    # generate result on test set size >= 10K sessions
    session_size_dict_large = {k: v for k, v in session_size_dict.items() if v >= 10000}
    print(f'session with test set size >= 10k, cnt={len(session_size_dict_large)}')
    all_df_large = all_df[all_df.task.isin(session_size_dict_large.keys())]
    agg_df_large = all_df_large.groupby('framework').agg(
        mean_auc=pd.NamedAgg(column='auc', aggfunc=np.mean),
        mean_sec_per_row=pd.NamedAgg(column='sec_per_row', aggfunc=np.mean),
        mean_genuine_sec_per_row=pd.NamedAgg(column='genuine_sec_per_row', aggfunc=np.mean),
    )
    print(agg_df_large)
    # generate result on test set size < 10K sessions
    session_size_dict_small = {k: v for k, v in session_size_dict.items() if v < 10000}
    print(f'session with test set size < 10k, cnt={len(session_size_dict_small)}')
    all_df_small = all_df[all_df.task.isin(session_size_dict_small.keys())]
    agg_df_small = all_df_small.groupby('framework').agg(
        mean_auc=pd.NamedAgg(column='auc', aggfunc=np.mean),
        mean_sec_per_row=pd.NamedAgg(column='sec_per_row', aggfunc=np.mean),
        mean_genuine_sec_per_row=pd.NamedAgg(column='genuine_sec_per_row', aggfunc=np.mean),
    )
    print(agg_df_small)
    
    # draw
    fig, axes = plt.subplots(3, 2, figsize=(13, 17), sharex=True, sharey=True,
                                        gridspec_kw={'hspace': 0.3})
    markers = {k: 'X' if k.startswith('F2S+') else 'o' for k in agg_df.index}
    markers['AGv053_high_tab'] = 'd'
    # ax0, ax1 with all sessions
    ax0 = axes[0, 0]
    ax0.grid()
    # ax0.set_xscale('log')
    ax0.tick_params(axis='both', which='both', labelbottom=True)
    ax0.set_title('CPP (4h8c)')
    sns.scatterplot(ax=ax0, data=agg_df, x='mean_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax0.legend(loc='lower right', ncol=2)
    ax1 = axes[0, 1]
    # ax1.set_xscale('log')
    ax1.grid()
    ax1.tick_params(axis='both', which='both', labelbottom=True, labelright=True)
    ax1.set_title('CPP (4h8c) resample to 10K')
    sns.scatterplot(ax=ax1, data=agg_df, x='mean_genuine_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax1.legend(loc='lower right', ncol=2)
    # ax2 with only large sessions
    ax2 = axes[1, 0]
    ax2.grid()
    # ax2.set_xscale('log')
    ax2.tick_params(axis='both', which='both', labelbottom=True)
    ax2.set_title('CPP 4 test_size>=10K sessions (4h8c)')
    sns.scatterplot(ax=ax2, data=agg_df_large, x='mean_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax2.legend(loc='lower right', ncol=2)
    ax3 = axes[1, 1]
    # ax3.set_xscale('log')
    ax3.grid()
    ax3.tick_params(axis='both', which='both', labelbottom=True, labelright=True)
    ax3.set_title('CPP 4 test>=10K sessions (4h8c), resample to 10K')
    sns.scatterplot(ax=ax3, data=agg_df_large, x='mean_genuine_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax3.legend(loc='lower right', ncol=2)
    # ax3 with only small sessions
    ax4 = axes[2, 0]
    ax4.grid()
    # ax4.set_xscale('log')
    ax4.set_title('CPP 56 test<10K sessions (4h8c)')
    sns.scatterplot(ax=ax4, data=agg_df_small, x='mean_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax4.legend(loc='lower right', ncol=2)
    ax5 = axes[2, 1]
    ax5.grid()
    # ax5.set_xscale('log')
    ax5.tick_params(axis='both', which='both', labelbottom=True, labelright=True)
    ax5.set_title('CPP 56 test<10K sessions (4h8c), resample to 10K')
    sns.scatterplot(ax=ax5, data=agg_df_small, x='mean_genuine_sec_per_row', y='mean_auc', hue='framework', s=80,
                    style='framework', markers=markers, alpha=0.75)
    ax5.legend(loc='lower right', ncol=2)
    # save to disk
    fig.savefig('cpp_report/Oct10.png')
    plt.close(fig)


if __name__ == '__main__':
    exp_result_dir = 'ExpResults/CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022'
    out_dir = 'cpp_report'
    # gather_all_ag_results(exp_result_dir, os.path.join(out_dir, 'CPP-Benchmark-n4dg.2xlarge-4h8c-Sep182022-all_results.csv'))

    # gather_all_cascade_results(exp_result_dir, out_dir, 'scores/cascade_results_2.csv')
    generate_report()