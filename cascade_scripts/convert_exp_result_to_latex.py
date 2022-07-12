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

Model_Name_Map_Best = OrderedDict([
    ('WeightedEnsemble_L3', ('WE\_L3', 'a')),
    ('WeightedEnsemble_L2', ('WE\_L2', 'a')),
    ('CatBoost_BAG_L1', ('Cat\_L1', 'd')),
    ('CatBoost_BAG_L2', ('Cat\_L2', 'd')),
    ('KNeighborsUnif_BAG_L1', ('KNN-U\_L1', 'e')),
    ('NeuralNetFastAI_BAG_L1', ('FAI\_L1', 'f')),
    ('NeuralNetFastAI_BAG_L2', ('FAI\_L2', 'f')),
    ('NeuralNetTorch_BAG_L1', ('NN\_L1', 'f')),
    ('NeuralNetTorch_BAG_L2', ('NN\_L2', 'f')),
    ('LightGBM_BAG_L1', ('GBM\_L1', 'g')),
    ('LightGBM_BAG_L2', ('GBM\_L2', 'g')),
    ('XGBoost_BAG_L1', ('XGB\_L1', 'g')),
    ('XGBoost_BAG_L2', ('XGB\_L2', 'g')),
    ('RandomForestGini_BAG_L1', ('RF-G\_L1', 'g')),
    ('RandomForestGini_BAG_L2', ('RF-G\_L2', 'g')),
    ('ExtraTreesGini_BAG_L1', ('XT-G\_L1', 'g')),
    ('ExtraTreesGini_BAG_L2', ('XT-G\_L2', 'g')),
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
    if '_MM' in args.exp_result_save_path:
        model_name_map = Model_Name_Map_MM
    elif 'best_quality' in args.exp_result_save_path:
        model_name_map = Model_Name_Map_Best
    else:
        model_name_map = Model_Name_Map
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
