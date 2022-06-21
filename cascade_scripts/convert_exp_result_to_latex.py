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

Model_Name_Map_MM = OrderedDict([
    ('WeightedEnsemble_L2', ('WE\_L2', 'a')),
    ('CatBoost', ('Cat', 'd')),
    ('VowpalWabbit', ('VW', 'd')),
    ('NeuralNetTorch', ('NN', 'f')),
    ('LightGBM', ('GBM', 'g')),
    ('XGBoost', ('XGB', 'g')),
    ('TextPredictor', ('TXT\_NN', 'h')),
    ('ImagePredictor', ('IMG\_NN', 'i'))
    ])


def main(args: argparse.Namespace):
    exp_df = pd.read_csv(args.exp_result_save_path).set_index('model')
    print(exp_df[[args.perf_metric_name,'speed']])
    output = []
    model_name_map = Model_Name_Map_MM if '_MM' in args.exp_result_save_path else Model_Name_Map
    for mname, (mname_short, fig_label) in model_name_map.items():
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
