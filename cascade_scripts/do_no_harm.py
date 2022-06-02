"""
Date: May 12, 2022
Author: Jiaying Lu
"""

import os
import time
from typing import List, Tuple, Dict, Optional
import argparse

# from sklearnex import patch_sklearn
# patch_sklearn()   # This cause bug!!!
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset, FeatureMetadata
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.metrics import accuracy, roc_auc
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.core.utils import generate_train_test_split
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import tqdm

from .cascade_utils import paretoset, rescale_by_pareto_frontier_model
from .cascade_utils import SPEED, ERROR, MODEL, SCORE
from .cascade_utils import get_exp_df_meta_columns

METRIC_FUNC_MAP = {'accuracy': accuracy, 'roc_auc': roc_auc}


def image_id_to_path(image_id: Optional[str]) -> Optional[str]:
    if isinstance(image_id, str):
        image_path = 'datasets/cpp_research_corpora/2021_60datasets_imgs_raw/' + image_id + '.jpg'
        if os.path.exists(image_path):
            return image_path
        else:
            return None
    else:
        return None


def get_cascade_metric_and_time_by_one_threshold(val_data: Tuple[np.ndarray, np.ndarray],
                                                 metric_name: str, cascade_threshold: float,
                                                 problem_type: str, num_classes: int,
                                                 cascade_model_seq: List[str],
                                                 model_pred_proba_dict: Dict[str, np.ndarray],
                                                 model_pred_time_dict: Dict[str, float],
                                                 ) -> Tuple[float, float]:
    # mimic logic here: https://github.com/awslabs/autogluon/blob/eb314b1032bc9bc3f611a4d6a0578370c4c89277/core/src/autogluon/core/trainer/abstract_trainer.py#L795
    global METRIC_FUNC_MAP
    X, Y = val_data[0], val_data[1]
    num_rows = Y.shape[0]
    if problem_type == BINARY:
        ret_pred_proba = np.zeros(num_rows, dtype='float32')
    elif problem_type == MULTICLASS:
        ret_pred_proba = np.zeros((num_rows, num_classes), dtype='float32')
    else:
        raise AssertionError(f'Invalid cascade problem_type: {problem_type}')
    ret_infer_time = 0.0
    accum_confident = np.zeros(num_rows, dtype=bool)
    for model_name in cascade_model_seq:
        pred_proba: np.ndarray = model_pred_proba_dict[model_name]
        # last_model just take over remaining unconfident rows
        if model_name == cascade_model_seq[-1]:
            unconfident = ~accum_confident
            ret_pred_proba[unconfident] = pred_proba[unconfident]
            ret_infer_time += (unconfident.sum() / num_rows * model_pred_time_dict[model_name])
            break
        if problem_type == BINARY:
            # if cascade_threshold is 0.5, will always exit in the first model
            confident = (pred_proba >= cascade_threshold) | (pred_proba <= (1-cascade_threshold))
        elif problem_type == MULTICLASS:
            confident = (pred_proba >= cascade_threshold).any(axis=1)
        else:
            raise AssertionError(f'Invalid cascade problem_type: {problem_type}')
        confident_to_add = ~accum_confident & confident
        ret_pred_proba[confident_to_add] = pred_proba[confident_to_add]
        # ret_infer_time += (confident_to_add.sum() / num_rows * model_pred_time_dict[model_name])
        ret_infer_time += (1.0 - accum_confident.sum() / num_rows) * model_pred_time_dict[model_name]
        accum_confident = accum_confident | confident
        if accum_confident.sum() >= num_rows:
            # exit cascade early
            # print(f'{cascade_threshold=}: After {model_name}, we collect all pred. Exit cascade')
            break
    metric_function = METRIC_FUNC_MAP[metric_name]
    metric_value = metric_function(y_true=Y, y_pred=ret_pred_proba)
    return (metric_value, ret_infer_time)


def simulate_refit_full_models_pred_proba_margin_time(predictor: TabularPredictor, 
        cascade_model_seq: List[str]) -> Tuple[dict]:
    predictor.persist_models('all')   # in order to get model objs
    model_pred_proba_dict: Dict[str, pd.DataFrame] = {}
    model_pred_time_dict: Dict[str, float] = {}   # margin time
    trainer = predictor._trainer
    as_multiclass: bool = trainer.problem_type == MULTICLASS
    leaderboard = predictor.leaderboard().set_index('model')
    for model_name in cascade_model_seq:
        # now assume each model is a _FULL model
        bag_model_name = model_name.replace('_FULL', '')
        model_pred_proba_dict[model_name] = predictor.get_oof_pred_proba(bag_model_name, as_multiclass=as_multiclass)
        bag_model_obj = trainer.models[bag_model_name]
        bag_model_cnt = len(bag_model_obj.models)
        bag_time_val_marginal = leaderboard.loc[bag_model_name].loc['pred_time_val_marginal']
        # print(f'{model_name}-{bag_model_name}: {bag_model_cnt=} {time_val_marginal=}')
        model_pred_time_dict[model_name] = bag_time_val_marginal / bag_model_cnt
    return model_pred_proba_dict, model_pred_time_dict


def hpo_one_param(predictor: TabularPredictor, leaderboard: pd.DataFrame, metric_name: str,
                  HPO_score_func_name: str = 'Rescale_Pareto',
                  allow_single_model: bool = True, 
                  cascade_model_seq: List[str] = []) -> Tuple[str, float]:
    """
    Args:
        metric_name: the metric name for calculating HPO score
        HPO_score_func_name: choices from ['HMean', 'Rescale_Pareto']
        allow_single_model: whether to allow return one single model (not using cascade) as return
    Returns:
        (model, threshold): 
          model: single model or cascade;
          float: threshold if chosen model is cascade; otherwise -1.0
    """
    # sanity check for input arguments
    if not cascade_model_seq:
        cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(leaderboard, predictor)
    if HPO_score_func_name == 'Rescale_Pareto':
        allow_single_model = True   # must allow, since it depends on Pareto Frontier
    # print(f'{cascade_model_seq=}')

    # get baseline using full cascade
    # models_fit_order: List[str] = leaderboard.sort_values('fit_order')['model'].tolist()   # this may only valid for non-bagging
    # last_model: str = models_fit_order[-1]   # "WeightedEnsemble_L2"
    last_model: str = predictor.get_model_best()
    trainer = predictor._trainer
    problem_type: str = trainer.problem_type   # BINARY or MULTICLASS
    val_data = predictor.load_data_internal('val')   # Tuple[np.array, np.array] for X, Y
    if val_data[0] is None and val_data[1] is None:
        model_pred_proba_dict, model_pred_time_marginal_dict = \
                simulate_refit_full_models_pred_proba_margin_time(predictor, cascade_model_seq)
        val_data = predictor.load_data_internal('train')   # oof_pred actually on train_data
        # print('bagging strategy')
        # print(f'{model_pred_time_marginal_dict=}')
        # print(f'{val_data[0].shape=}, {val_data[1].shape=}')
        # print(f'{model_pred_proba_dict["RandomForestGini_BAG_L2_FULL"].shape=}')
    else:
        # models Not use bagging strategy
        # run all models on val data in just one time, keep everything in-record
        # this returns **marginal** time
        model_pred_proba_dict, model_pred_time_marginal_dict = \
                trainer.get_model_pred_proba_dict(val_data[0], models=cascade_model_seq, record_pred_time=True)
    # print(f'get {model_pred_proba_dict=}, {model_pred_time_marginal_dict=}')
    # print(f'{last_model=}')
    full_cascade_time = sum(model_pred_time_marginal_dict.values())
    full_cascade_metric_value = METRIC_FUNC_MAP[metric_name](y_true=val_data[1], y_pred=model_pred_proba_dict[last_model])
    num_classes = trainer.num_classes
    # do grid search
    max_HPO_score = 0.0
    cascade_perf_inftime_l = []   # model_name, performance, infer_time
    for threshold in tqdm.tqdm([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.825, 0.85, 0.875, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]):
        metric_value, infer_time = get_cascade_metric_and_time_by_one_threshold(val_data, metric_name, threshold,
                problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict)
        cascade_perf_inftime_l.append([f'cascade-{threshold}', metric_value, infer_time])
    PRED_TIME = 'pred_time_val'
    PERFORMANCE = 'score_val'
    columns = [MODEL, PERFORMANCE, PRED_TIME]
    cascade_perf_inftime_l = pd.DataFrame(cascade_perf_inftime_l, columns=columns)
    model_perf_inftime_df = pd.concat([leaderboard[columns], cascade_perf_inftime_l]).set_index(MODEL)
    max_HPO_score = 0.0
    chosen_model = ''
    if HPO_score_func_name == 'HMean':
        # harmonic mean of infer time and val metric
        HPO_score_func = lambda a, b: 2.0 / (1.0/a + 1.0/b)
        for i in range(model_perf_inftime_df.shape[0]):
            row = model_perf_inftime_df.iloc[i]
            if not allow_single_model and 'cascade' not in row.name:
                continue
            performance = row[PERFORMANCE] / full_cascade_metric_value
            speed = full_cascade_time / row[PRED_TIME]
            HPO_score = HPO_score_func(performance, np.log10(speed))
            if HPO_score >= max_HPO_score:
                max_HPO_score = HPO_score
                chosen_model = row.name
    elif HPO_score_func_name == 'Rescale_Pareto':
        # get speed in terms of X rows/second
        speed_soft_cap = 1000
        weights = (-1.0, 0.01)
        model_perf_inftime_df[SPEED] = val_data[0].shape[0] / model_perf_inftime_df[PRED_TIME] 
        pareto_mask = paretoset(model_perf_inftime_df[[PERFORMANCE, SPEED]], sense=['max', 'max'])
        model_perf_inftime_df = model_perf_inftime_df[pareto_mask]
        # currently only ocnsider ACC and ROC_AUC
        model_perf_inftime_df[ERROR] = 1.0 - model_perf_inftime_df[PERFORMANCE]
        # set up random guess baseline performance
        if problem_type == MULTICLASS and predictor.eval_metric.name == 'accuracy':
            # use major class percentage as baseline performance
            random_guess_perf = val_data[1].value_counts(normalize=True).max()
        else:
            # use preset random guess model
            random_guess_perf = None
        # print(f'{random_guess_perf=}, {predictor.eval_metric.name=}')
        model_perf_inftime_df = rescale_by_pareto_frontier_model(model_perf_inftime_df, 
                speed_soft_cap=speed_soft_cap, weights=weights,
                metric_name=predictor.eval_metric.name, random_guess_perf=random_guess_perf)
        model_perf_inftime_df = model_perf_inftime_df.sort_values(by=SCORE, ascending=False)
        chosen_model = model_perf_inftime_df.iloc[0].name
    else:
        raise ValueError(f"Invalid Input Arg {HPO_score_func_name=} for hpo_one_param()")
    # print(f'{chosen_model=}, {max_HPO_score}')
    if 'cascade' in chosen_model:
        chosen_model_name = 'cascade'
        chosen_threshold = float(chosen_model.split('-')[1])
    else:
        chosen_model_name = chosen_model
        chosen_threshold = None
    return (chosen_model_name, chosen_threshold)


def get_partial_weighted_ensemble_name(predictor: TabularPredictor, base_models: List[str]) -> str:
    assert len(base_models) > 0
    base_models_set = set(base_models)
    for successor in predictor._trainer.model_graph.successors(base_models[0]):
        precessors = set(predictor._trainer.model_graph.predecessors(successor))
        if precessors == base_models_set:
            return successor
    # reach here means no weighted ensemble for current base_models
    successor = predictor.fit_weighted_ensemble(base_models, name_suffix='_Cascade')[0]
    return successor


def get_cascade_model_sequence_by_val_marginal_time(leaderboard: pd.DataFrame, predictor: TabularPredictor,
                                                    are_member_of_best: bool = True,
                                                    better_than_prev: bool = True,
                                                    add_partial_weighted_ensemble: bool = False) -> List[str]:
    """
        Args:
            leaderboard: stores info about models produced during fit()
            are_member_of_best: whether to constrain output models must be members of best_model
            better_than_prev: the Pareto heuristic that we only include a model in the cascade if has a higher accuracy than any of the models earlier in the cascade
    """
    # Rule1: from fast to slow
    # Rule2: Layer by Layer
    leaderboard_sorted = leaderboard.sort_values(['stack_level', 'pred_time_val_marginal'], ascending=[True, True])
    model_sequence = leaderboard_sorted['model'].tolist()
    if better_than_prev:
        val_scores = leaderboard_sorted['score_val'].tolist()
        tmp = []
        max_val_score = 0.0
        for val_score, model_name in zip(val_scores, model_sequence):
            if val_score < max_val_score:
                continue
            max_val_score = val_score
            tmp.append(model_name)
        model_sequence = tmp
    if are_member_of_best:
        valid_cascade_models = predictor._trainer.get_minimum_model_set(predictor.get_model_best())
        model_sequence = [m for m in model_sequence if m in valid_cascade_models]
    if add_partial_weighted_ensemble:
        model_seq_length = len(model_sequence)
        # print(model_seq_length)
        for i in range(model_seq_length-2, 0, -1):
            partial_model_seq = model_sequence[:i]
            # print(f'{i=}, {partial_model_seq=}')
            partial_we_model = get_partial_weighted_ensemble_name(predictor, partial_model_seq) 
            model_sequence.insert(i, partial_we_model)
    # print(f'final, {model_sequence=}')
    return model_sequence


def main(args: argparse.Namespace):
    dataset_name = args.dataset_name
    model_save_path = args.model_save_path
    do_multimodal = args.do_multimodal
    force_training = args.force_training
    run_xtime = args.run_xtime
    exp_result_save_path = args.exp_result_save_path
    presets = args.predictor_presets
    ndigits = 4
   
    # Cover Type MultiClass
    if dataset_name == 'CoverTypeMulti':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification/'
        label = 'Cover_Type'
        path_train = path_prefix + 'train_data.csv'
        path_test = path_prefix + 'test_data.csv'
        eval_metric = 'accuracy'
        model_hyperparameters = 'default'
    # Adult Income Dataset
    elif dataset_name == 'Inc':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
        label = 'class'
        path_train = path_prefix + 'train.csv'
        path_test = path_prefix + 'test.csv'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
    elif dataset_name == 'CPP-6aa99d1a':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/6aa99d1a-1d4b-4d30-bd8b-a26f259b6482/'
        label = 'label'
        path_train = path_prefix + 'train/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
    elif dataset_name == 'CPP-3564a7a7':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/3564a7a7-0e7c-470f-8f9e-5a029be8e616/'
        label = 'label'
        path_train = path_prefix + 'train/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
    else:
        print(f'currently not support dataset_name={dataset_name}')
        exit(-1)

    train_data = TabularDataset(path_train)
    test_data = TabularDataset(path_test)

    fit_kwargs = dict(
        train_data=train_data,
        hyperparameters=model_hyperparameters,
        presets=presets,
    )
    if do_multimodal:
        # update several fit kwargs
        hyperparameters = get_hyperparameter_config('multimodal')
        image_col = 'image_id'
        train_data[image_col] = train_data[image_col].apply(image_id_to_path)
        test_data[image_col] = test_data[image_col].apply(image_id_to_path)
        feature_metadata = FeatureMetadata.from_df(train_data)
        feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
        fit_kwargs['hyperparameters'] = hyperparameters
        fit_kwargs['feature_metadata'] = feature_metadata
        fit_kwargs['presets'] = None
    
    if force_training is True or (not os.path.exists(model_save_path)):
        predictor = TabularPredictor(
            label=label,
            eval_metric=eval_metric,
            path=model_save_path,
        )
        predictor.fit(**fit_kwargs)
    else:
        predictor = TabularPredictor.load(model_save_path)
    predictor.persist_models('all')   # to speed-up
    leaderboard = predictor.leaderboard(test_data)
    # Load or Create Exp Result df
    meta_cols = get_exp_df_meta_columns(predictor._learner.problem_type)
    if os.path.exists(exp_result_save_path):
        exp_result_df = pd.read_csv(exp_result_save_path, index_col='model')
    else:
        exp_result_df = pd.DataFrame(columns=meta_cols).set_index('model').dropna()

    # Infer with each single model of AG stack ensemble
    print('--------')
    best_model = predictor.get_model_best()
    best_model_members = predictor._trainer.get_minimum_model_set(best_model, include_self=False)
    print(f'AG0.4 with best_model={best_model}--{best_model_members}:')
    for model_name, can_infer in zip(leaderboard['model'], leaderboard['can_infer']):
        if model_name.endswith('_Cascade'):
            continue
        if not can_infer:
            continue
        infer_times = []
        for _ in range(run_xtime):
            ts = time.time()
            pred_proba = predictor.predict_proba(test_data, model=model_name)
            te = time.time()
            infer_times.append(te-ts)
        infer_times = sum(infer_times) / len(infer_times)
        test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
        speed = test_data.shape[0] / infer_times
        test_metric1 = test_metrics[meta_cols[-2]]
        test_metric2 = test_metrics[meta_cols[-1]]
        exp_result_df.loc[model_name] = [infer_times, speed, test_metric1, test_metric2]
        if model_name == best_model:
            print(f'{model_name}: {test_metrics} | time: {infer_times}s')
    print('--------')

    """
    # Infer Time for fast to slow approach F2SP
    print('--------')
    model_name = 'F2SP'
    # Step1: get the model sequence by val inference
    cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(leaderboard, predictor)
    print(f'{model_name}: cascade_model_seq={cascade_model_seq}, default_threshol=0.9')
    # Step2: stop can be done after each model
    infer_times = []
    for _ in range(run_xtime):
        ts = time.time()
        pred_proba = predictor.predict_proba(test_data, model=cascade_model_seq)
        te = time.time()
        infer_times.append(te-ts)
    infer_times = sum(infer_times) / len(infer_times)
    speed = test_data.shape[0] / infer_times
    test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
    test_metric1 = test_metrics[meta_cols[-2]]
    test_metric2 = test_metrics[meta_cols[-1]]
    exp_result_df.loc[model_name] = [infer_times, speed, test_metric1, test_metric2]
    print(f'{model_name}: {test_metrics} | time: {infer_times}s')
    print('--------')
    """

    """
    # TODO: in progress
    # Infer Time for fast to slow approach F2SP++
    # Step1: get the model sequence by val inference
    cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(leaderboard, predictor, add_partial_weighted_ensemble=True)
    # Step2: stop can be done after each model
    ts = time.time()
    pred_proba = predictor.predict_proba(test_data, model=cascade_model_seq)
    te = time.time()
    print('--------')
    print('F2SP++:')
    print(f'cascade_model_seq={cascade_model_seq}, default_threshol=0.9')
    print(predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True))
    print(f'{te - ts}s')
    print('--------')
    """

    """
    # Infer Time for F2SP w/ HPO
    print('--------')
    model_name = 'F2SP/T(RePrt)'
    print('F2SP w/ HPO (Rescale_Pareto):')
    cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(leaderboard, predictor)
    chosen_model_name, chosen_threshold = hpo_one_param(predictor, leaderboard, eval_metric, 
            'Rescale_Pareto', False, cascade_model_seq)
    infer_times = []
    for _ in range(run_xtime):
        if chosen_model_name == 'cascade':
            ts = time.time()
            learner = predictor._learner
            trainer = predictor._trainer
            test_data_X = learner.transform_features(test_data)
            model_pred_proba_dict = trainer.get_model_pred_proba_dict(
                    test_data_X, cascade_model_seq, fit=False,
                    cascade=True, cascade_threshold=chosen_threshold
                    )
            pred_proba = model_pred_proba_dict[cascade_model_seq[-1]]
            if learner.problem_type == BINARY:
                pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(pred_proba)
            te = time.time()
        else:
            ts = time.time()
            pred_proba = predictor.predict_proba(test_data, model=chosen_model_name)
            te = time.time()
        infer_times.append(te-ts)
    infer_times = sum(infer_times) / len(infer_times)
    speed = test_data.shape[0] / infer_times
    test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
    test_metric1 = test_metrics[meta_cols[-2]]
    test_metric2 = test_metrics[meta_cols[-1]]
    exp_result_df.loc[model_name] = [infer_times, speed, test_metric1, test_metric2]
    print(f'{model_name}: cascade_model_seq={cascade_model_seq}, chosen_threshold={chosen_model_name, chosen_threshold}')
    print(f'{model_name}: {test_metrics} | time: {infer_times}s')
    """

    # store exp_result_df into disk
    print(exp_result_df.round(ndigits))
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
    parser.add_argument('--run_xtime', type=int, default=3, help="Run X times of each approach to get stable infer time.")
    parser.add_argument('--predictor_presets', type=str, default='medium_quality')
    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)
