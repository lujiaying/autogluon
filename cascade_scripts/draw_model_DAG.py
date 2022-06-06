"""
Date: June 6, 2022
Author: Jiaying Lu
"""
import argparse

from autogluon.tabular import TabularPredictor
import networkx as nx


def main(args: argparse.Namespace):
    predictor = TabularPredictor.load(args.model_save_path)
    DAG = predictor._trainer.model_graph
    outpath = f'{args.model_save_path}/DAG.dot'
    nx.drawing.nx_agraph.write_dot(DAG, outpath)
    print(f'[Success] Generate dot file into {outpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments for Drawing/Exporting Model DAG to Dot File")
    parser.add_argument('--model_save_path', type=str, required=True)

    args = parser.parse_args()
    print(f'In arguments: {args}')

    main(args)
