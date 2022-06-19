"""
Date: Jun 19, 2022
Author: Jiaying Lu
"""

from collections import OrderedDict
import argparse

import pandas as pd


Model_Name_Map = OrderedDict([
    ('WeightedEnsemble_L3', ('WE\_L3', 'a')),
    ('WeightedEnsemble_L2', ('WE\_L2', 'a')),
    ('CatBoost', ('Cat', 'd')),
    ('KNeighborsUnif', ('KNN-U', 'e')),
    ('NeuralNetFastAI', ('FAI', 'f')),
    ('NeuralNetTorch', ('NN', 'f')),
    ('LightGBM', ('GBM', 'g')),
    ('XGBoost', ('XGB', 'g')),
    ('RandomForestGini', ('RF-G', 'g')),
    ('ExtraTreesGini', ('XT-G', 'g'))
    ])


def main(args: argparse.Namespace):
    exp_df = pd.read_csv(args.exp_result_save_path).set_index('model')
    print(exp_df[[args.perf_metric_name,'speed']])
    output = []
    for mname, (mname_short, fig_label) in Model_Name_Map.items():
        if mname not in exp_df.index:
            continue
        row = exp_df.loc[mname]
        performance = round(row[args.perf_metric_name], 4)
        speed = round(row['speed'], 1)
        output.append((performance, speed, fig_label, mname_short))
    output_df = pd.DataFrame(output, columns=['x', 'y', 'label', 'method']) 
    print(output_df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="output data in Latex format for figure drawing")
    parser.add_argument('--exp_result_save_path', type=str, required=True)
    parser.add_argument('--perf_metric_name', type=str, required=True)
    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)
