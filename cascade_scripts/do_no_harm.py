"""
Date: May 12, 2022
Author: Jiaying Lu
"""

import os
import time
from typing import List, Tuple, Dict, Optional, Union, Set, Any
from functools import partial
import itertools
import argparse


# from sklearnex import patch_sklearn
# patch_sklearn()   # This cause bug!!!
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor, TabularDataset, FeatureMetadata
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.metrics import accuracy, roc_auc
from autogluon.core.data.label_cleaner import LabelCleanerMulticlassToBinary
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
import tqdm

from .cascade_utils import load_dataset
from .cascade_utils import paretoset, rescale_by_pareto_frontier_model
from .cascade_utils import SPEED, ERROR, MODEL, SCORE
from .cascade_utils import get_exp_df_meta_columns, MAIN_METRIC_COL, SEC_METRIC_COL

METRIC_FUNC_MAP = {'accuracy': accuracy, 'acc': accuracy, 'roc_auc': roc_auc}
THRESHOLDS_BINARY = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0]
# generate from softmax(norm.pdf(thresholds, loc=0.9, scale=0.25)
PROB_BINARY = [0.03932009, 0.05486066, 0.08038427, 0.1100718 , 0.12056911, 0.12443972, 0.12345325, 0.12056911, 0.11600155, 0.11033044]
# generate from softmax(norm.pdf(thresholds, loc=0.75, scale=0.25)
THRESHOLDS_MULTICLASS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
PROB_MULTICLASS = [0.02542674, 0.02637472, 0.03100157, 0.04546687, 0.06575639,
       0.09472447, 0.11937201, 0.1232042 , 0.11937201, 0.10897893,
       0.09472447, 0.07958616, 0.06601146]
PRED_TIME = 'pred_time_val'
PERFORMANCE = 'score_val'
PWE_suffix = '_PWECascade'
COLS_REPrt = [MODEL, PERFORMANCE, PRED_TIME]       # columns for rescale_by_pareto_frontier_model()


def image_id_to_path(image_id: Optional[str], image_path: str, image_path_suffix: str = ''
        ) -> Optional[str]:
    if isinstance(image_id, str):
        image_path = image_path + image_id + image_path_suffix
        if os.path.exists(image_path):
            return image_path
        else:
            return None
    else:
        return None

image_id_to_path_cpp = partial(image_id_to_path, 'datasets/cpp_research_corpora/2021_60datasets_imgs_raw/', 'jpg')
image_id_to_path_petfinder = partial(image_id_to_path, 'datasets/petfinder_processed/', '')


def get_cascade_metric_and_time_by_threshold(val_data: Tuple[np.ndarray, np.ndarray],
                                             metric_name: str, 
                                             cascade_thresholds: Union[float, List[float]],
                                             problem_type: str, num_classes: int,
                                             cascade_model_seq: List[str],
                                             model_pred_proba_dict: Dict[str, np.ndarray],
                                             model_pred_time_dict: Dict[str, float],
                                             predictor: TabularPredictor,
                                             ) -> Tuple[float, float]:
    # mimic logic here: https://github.com/awslabs/autogluon/blob/eb314b1032bc9bc3f611a4d6a0578370c4c89277/core/src/autogluon/core/trainer/abstract_trainer.py#L795
    global METRIC_FUNC_MAP
    if isinstance(cascade_thresholds, float):
        cascade_thresholds = [cascade_thresholds for _ in range(len(cascade_model_seq)-1)]
    elif isinstance(cascade_thresholds, tuple):
        cascade_thresholds = list(cascade_thresholds)
    elif isinstance(cascade_thresholds, np.ndarray):
        cascade_thresholds = cascade_thresholds.tolist()
    # add a dummy threshold for last model in cascade_model_seq
    assert len(cascade_thresholds) == len(cascade_model_seq) - 1
    cascade_thresholds: list = cascade_thresholds.copy()
    cascade_thresholds.append(None)
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
    executed_model_names = set()
    # model#1 can be XG_L1, #2 can be Cat_L2
    # so simulation need to consider all children infer time
    for model_name, threshold in zip(cascade_model_seq, cascade_thresholds):
        pred_proba: np.ndarray = model_pred_proba_dict[model_name]
        # last_model just take over remaining unconfident rows
        if model_name == cascade_model_seq[-1]:
            unconfident = ~accum_confident
            ret_pred_proba[unconfident] = pred_proba[unconfident]
            nonexec_marginal_time_table = get_non_excuted_predecessors_marginal_time(predictor, model_name, executed_model_names, model_pred_time_dict)
            ret_infer_time += (unconfident.sum() / num_rows * nonexec_marginal_time_table)
            # ret_infer_time += (unconfident.sum() / num_rows * model_pred_time_dict[model_name])
            break
        if problem_type == BINARY:
            confident = (pred_proba >= threshold) | (pred_proba <= (1-threshold))
        elif problem_type == MULTICLASS:
            confident = (pred_proba >= threshold).any(axis=1)
        else:
            raise AssertionError(f'Invalid cascade problem_type: {problem_type}')
        confident_to_add = ~accum_confident & confident
        ret_pred_proba[confident_to_add] = pred_proba[confident_to_add]
        nonexec_marginal_time_table = get_non_excuted_predecessors_marginal_time(predictor, model_name, executed_model_names, model_pred_time_dict)
        ret_infer_time += (1.0 - accum_confident.sum() / num_rows) * nonexec_marginal_time_table 
        # ret_infer_time += (1.0 - accum_confident.sum() / num_rows) * model_pred_time_dict[model_name]
        accum_confident = accum_confident | confident
        if accum_confident.sum() >= num_rows:
            # exit cascade early
            # print(f'{cascade_threshold=}: After {model_name}, we collect all pred. Exit cascade')
            break
    metric_function = METRIC_FUNC_MAP[metric_name]
    metric_value = metric_function(y_true=Y, y_pred=ret_pred_proba)
    return (metric_value, ret_infer_time)


def simulate_bag_models_pred_proba_margin_time(predictor: TabularPredictor, 
        cascade_model_seq: List[str]) -> Tuple[dict, dict]:
    model_pred_proba_dict: Dict[str, pd.DataFrame] = {}
    model_pred_time_dict: Dict[str, float] = {}   # margin time
    trainer = predictor._trainer
    as_multiclass: bool = trainer.problem_type == MULTICLASS
    leaderboard = predictor.leaderboard(silent=True).set_index(MODEL)
    for model_name in cascade_model_seq:
        # TODO: to cover high_quality refit models contains _FULL str
        # now assume each model contains oof_pred_proba
        if '_FULL' in model_name:
            raise ValueError(f'{model_name} is a refit model. Simulation on high_quality preset is not implemented yet')
        """
        bag_model_name = model_name.replace('_FULL', '')
        model_pred_proba_dict[model_name] = predictor.get_oof_pred_proba(bag_model_name, as_multiclass=as_multiclass)
        bag_model_obj = trainer.models[bag_model_name]
        bag_model_cnt = len(bag_model_obj.models)
        bag_time_val_marginal = leaderboard.loc[bag_model_name].loc['pred_time_val_marginal']
        # print(f'{model_name}-{bag_model_name}: {bag_model_cnt=} {time_val_marginal=}')
        model_pred_time_dict[model_name] = bag_time_val_marginal / bag_model_cnt
        """
        model_pred_proba_dict[model_name] = predictor.get_oof_pred_proba(model_name, as_multiclass=as_multiclass)
        model_pred_time_dict[model_name] = leaderboard.loc[model_name].loc['pred_time_val_marginal']
    return model_pred_proba_dict, model_pred_time_dict


def helper_get_val_data(predictor: TabularPredictor) -> Tuple[Tuple[np.ndarray, np.ndarray], bool]:
    """
    For models trained with bagging strategy, 
    we no longer able to directly get val_data
    """
    val_data = predictor.load_data_internal('val')   # Tuple[np.array, np.array] for X, Y
    is_trained_bagging = False
    if val_data[0] is None and val_data[1] is None:
        val_data = predictor.load_data_internal('train')   # oof_pred actually on train_data
        is_trained_bagging = True
    return val_data, is_trained_bagging


def get_models_pred_proba_on_val(predictor: TabularPredictor, cascade_model_seq: List[str]
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    trainer = predictor._trainer
    val_data, is_trained_bagging = helper_get_val_data(predictor)
    if is_trained_bagging is True:
        # This branch covers best-quality presets
        # TODO: to cover refit_full models (high-quality presets)
        model_pred_proba_dict, model_pred_time_marginal_dict = \
                simulate_bag_models_pred_proba_margin_time(predictor, cascade_model_seq)
    else:
        # models Not use bagging strategy
        # run all models on val data in just one time, keep everything in-record
        # this returns **marginal** time
        model_pred_proba_dict, model_pred_time_marginal_dict = \
                trainer.get_model_pred_proba_dict(val_data[0], models=cascade_model_seq, record_pred_time=True)
    return model_pred_proba_dict, model_pred_time_marginal_dict, val_data


def wrap_rescale_by_pareto_frontier_model(model_perf_inftime_df: pd.DataFrame, val_data: tuple,
        predictor: TabularPredictor, mask_pareto_dominated_models: bool = False) -> pd.DataFrame:
    """
    This is a wrapper for rescale_by_pareto_frontier_model()
    which contains essential presets/hyperparameters
    """
    model_perf_inftime_df = model_perf_inftime_df.copy()
    problem_type = predictor._learner.problem_type
    speed_soft_cap = 1000   # rows/second
    weights = (-1.0, 0.01)
    # get speed in terms of X rows/second
    model_perf_inftime_df[SPEED] = val_data[0].shape[0] / model_perf_inftime_df[PRED_TIME] 
    if mask_pareto_dominated_models:
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
    return model_perf_inftime_df


def choose_best_threshold_model_by_Rescale_Pareto(model_perf_inftime_df: pd.DataFrame, val_data: tuple,
        predictor: TabularPredictor) -> Tuple[str, float, float]:
    model_perf_inftime_df = wrap_rescale_by_pareto_frontier_model(model_perf_inftime_df, val_data, predictor)
    model_perf_inftime_df = model_perf_inftime_df.sort_values(by=SCORE, ascending=False)
    chosen_row = model_perf_inftime_df.iloc[0]
    chosen_model = chosen_row.name   # 'cascade-0.7_0.8_0.9'
    return chosen_model, chosen_row.loc[PRED_TIME], chosen_row.loc[PERFORMANCE]


def hpo_one_param(predictor: TabularPredictor, metric_name: str,
                  HPO_score_func_name: str = 'Rescale_Pareto',
                  allow_single_model: bool = True, 
                  cascade_model_seq: List[str] = []) -> Tuple[str, Optional[float], float, float]:
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
        cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(predictor)
    if HPO_score_func_name == 'Rescale_Pareto':
        allow_single_model = True   # must allow, since it depends on Pareto Frontier
    # print(f'{cascade_model_seq=}')
    leaderboard = predictor.leaderboard(silent=True)

    # get baseline using full cascade
    cascade_model_all_predecessors = get_all_predecessor_model_names(predictor, cascade_model_seq, include_self=True)
    model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
            get_models_pred_proba_on_val(predictor, cascade_model_all_predecessors)
    trainer = predictor._trainer
    problem_type: str = trainer.problem_type   # BINARY or MULTICLASS
    num_classes = trainer.num_classes
    # do grid search
    cascade_perf_inftime_l = []   # model_name, performance, infer_time
    threshold_candidate = THRESHOLDS_BINARY if problem_type == BINARY else THRESHOLDS_MULTICLASS
    for threshold in tqdm.tqdm(threshold_candidate):
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, metric_name, threshold,
                problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
        #TODO: change starting str into a variable
        cascade_perf_inftime_l.append([f'cascade-{threshold}', metric_value, infer_time])
    columns = [MODEL, PERFORMANCE, PRED_TIME]
    cascade_perf_inftime_l = pd.DataFrame(cascade_perf_inftime_l, columns=columns)
    model_perf_inftime_df = pd.concat([leaderboard[columns], cascade_perf_inftime_l]).set_index(MODEL)
    chosen_model = ''
    if HPO_score_func_name == 'HMean':
        max_HPO_score = 0.0
        last_model: str = cascade_model_seq[-1]
        full_cascade_time = sum(model_pred_time_marginal_dict.values())
        full_cascade_metric_value = METRIC_FUNC_MAP[metric_name](y_true=val_data[1], y_pred=model_pred_proba_dict[last_model])
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
        raise ValueError('Should not enter this branch in hpo_one_param()')
    elif HPO_score_func_name == 'Rescale_Pareto':
        chosen_model, time_val, score_val = choose_best_threshold_model_by_Rescale_Pareto(model_perf_inftime_df, val_data, predictor)
    else:
        raise ValueError(f"Invalid Input Arg {HPO_score_func_name=} for hpo_one_param()")
    # print(f'{chosen_model=}, {max_HPO_score}')
    if 'cascade' in chosen_model:
        chosen_model_name = 'cascade'
        chosen_threshold = float(chosen_model.split('-')[1])
    else:
        chosen_model_name = chosen_model
        chosen_threshold = None
    return chosen_model_name, chosen_threshold, time_val, score_val


def hpo_multi_params_random_search(predictor: TabularPredictor, metric_name: str, 
        cascade_model_seq: List[str],
        HPO_score_func_name: str = 'Rescale_Pareto',
        num_trails: int = 1000,
        ) -> Tuple[str, Optional[List[float]], float, float]:
    """
    Conduct a randommized search over hyperparameters
    """
    model_delim = '-'
    cas_delim = '_'
    cas_starting_str = 'CascadeThreshold'
    rng = np.random.default_rng(0)
    cascade_len = len(cascade_model_seq)
    problem_type = predictor._learner.problem_type
    leaderboard = predictor.leaderboard(silent=True)
    if problem_type == BINARY:
        thresholds_cands = THRESHOLDS_BINARY
        thresholds_probs = PROB_BINARY
    elif problem_type == MULTICLASS:
        thresholds_cands = THRESHOLDS_MULTICLASS
        thresholds_probs = PROB_MULTICLASS
    else:
        raise ValueError(f'Not support problem_type={problem_type}')

    # Get val pred proba
    cascade_model_all_predecessors = get_all_predecessor_model_names(predictor, cascade_model_seq, include_self=True)
    model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
            get_models_pred_proba_on_val(predictor, list(cascade_model_all_predecessors))
    # Get HPO score using one set of sampled thresholds
    num_classes = predictor._trainer.num_classes
    cascade_perf_inftime_l = []   # model_name, performance, infer_time
    if len(thresholds_cands)**(cascade_len-1) <= num_trails:
        # downgrade to exhaustive search because search space is accountable
        search_space = itertools.product(thresholds_cands, repeat=cascade_len-1)
    else:
        search_space = rng.choice(thresholds_cands, size=(num_trails, cascade_len-1), p=thresholds_probs).tolist()
    print('[hpo_multi_params_random_search] Start produce val_metrics, and val_time over search space')
    for thresholds in tqdm.tqdm(search_space):
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, metric_name, thresholds,
                problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
        thresholds_str = cas_delim.join(map(str, thresholds))
        cascade_perf_inftime_l.append([f'{cas_starting_str}{model_delim}{thresholds_str}', metric_value, infer_time])
    global COLS_REPrt
    columns = COLS_REPrt
    cascade_perf_inftime_l = pd.DataFrame(cascade_perf_inftime_l, columns=columns)
    model_perf_inftime_df = pd.concat([leaderboard[columns], cascade_perf_inftime_l]).set_index(MODEL)
    if HPO_score_func_name == 'Rescale_Pareto':
        chosen_model, time_val, score_val = choose_best_threshold_model_by_Rescale_Pareto(model_perf_inftime_df, val_data, predictor)
    else:
        raise ValueError(f'hpo_multi_params_random_search() not support arg HPO_score_func_name={HPO_score_func_name}')
    if chosen_model.startswith(cas_starting_str):
        chosen_model_name = 'cascade'
        chosen_threshold = chosen_model.split(model_delim)[1].split(cas_delim)
        chosen_threshold = list(map(float, chosen_threshold))
    else:
        chosen_model_name = chosen_model
        chosen_threshold = None
    return chosen_model_name, chosen_threshold, time_val, score_val


def hpo_multi_params_TPE(predictor: TabularPredictor, metric_name: str, 
        cascade_model_seq: List[str],
        HPO_score_func_name: str = 'Rescale_Pareto',
        num_trails: int = 500,
        ) -> Tuple[str, Optional[List[float]], float, float]:
    """
    Use TPE for HP
    Currently use hyperopt implementation
    """
    def _wrapper_obj_fn(model_perf_inftime_df: pd.DataFrame,
            val_data: Tuple[np.ndarray, np.ndarray], metric_name: str,
            problem_type: str, num_classes: int,
            cascade_model_seq: List[str],
            model_pred_proba_dict: Dict[str, np.ndarray],
            model_pred_time_dict: Dict[str, float],
            predictor: TabularPredictor,
            hyparams: Dict[str, Any],
            ) -> float:
        """
        Returns a loss that we want to minimize
        """
        # cas_ts_try: Dict[int, float],
        cas_ts_try = hyparams['thresholds']
        cas_ts_try_list = [0.0 for _ in range(len(cas_ts_try))]
        for idx, ts in cas_ts_try.items():
            cas_ts_try_list[idx] = ts
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, metric_name, cas_ts_try_list,
                problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_dict, predictor)
        cascade_try_name = 'CascadeThreshold_trial'
        model_perf_inftime_df = model_perf_inftime_df.copy()
        model_perf_inftime_df.loc[cascade_try_name] = [metric_value, infer_time]
        model_perf_inftime_df = wrap_rescale_by_pareto_frontier_model(model_perf_inftime_df, val_data, predictor)
        goodness_score = model_perf_inftime_df.loc[cascade_try_name][SCORE]
        return -goodness_score

    from hyperopt import fmin, tpe, hp
    if HPO_score_func_name != 'Rescale_Pareto':
        raise ValueError('Currently hpo_multi_params_TPE() only support "Rescale_Pareto" as goodness function')
    cascade_len = len(cascade_model_seq)
    problem_type = predictor._learner.problem_type
    leaderboard = predictor.leaderboard(silent=True)
    num_classes = predictor._trainer.num_classes
    if problem_type == BINARY:
        ts_min = 0.5
        ts_min_mild = 0.75
        ts_max = 1.0
    elif problem_type == MULTICLASS:
        ts_min = 0.0
        ts_min_mild = 0.33
        ts_max = 1.0
    else:
        raise ValueError(f'Not support problem_type={problem_type}')

    # Get val pred proba
    cascade_model_all_predecessors = get_all_predecessor_model_names(predictor, cascade_model_seq, include_self=True)
    model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
            get_models_pred_proba_on_val(predictor, list(cascade_model_all_predecessors))
    global COLS_REPrt
    model_perf_inftime_df = leaderboard[COLS_REPrt].copy().set_index(MODEL)
    # start the HPO
    object_func = partial(_wrapper_obj_fn, model_perf_inftime_df, val_data, metric_name, problem_type, num_classes, 
            cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
    search_space = hp.choice('cascade_thresholds', [
        {
            'mode': 'wild',
            'thresholds': {i: hp.uniform(f'wild_t_{i}', ts_min, ts_max) for i in range(cascade_len-1)},
        },
        {
            'mode': 'mild',
            'thresholds': {i: hp.uniform(f'mild_t_{i}', ts_min_mild, ts_max) for i in range(cascade_len-1)},
        },
    ])
    best = fmin(object_func,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trails)
    best_thresholds = [0.0 for _ in range(cascade_len-1)]
    for k, v in best.items():
        if k == 'cascade_thresholds':
            continue
        idx = int(k.split('_')[-1])
        best_thresholds[idx] = v
    score_val, time_val = get_cascade_metric_and_time_by_threshold(val_data, metric_name, best_thresholds,
            problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
    model_perf_inftime_df.loc['cascade'] = [score_val, time_val]
    chosen_model, time_val, score_val = choose_best_threshold_model_by_Rescale_Pareto(model_perf_inftime_df, val_data, predictor)
    if chosen_model != 'cascade':
        chosen_thresholds = None
    else:
        chosen_thresholds = best_thresholds
    print(f'[DEBUG] TPE got best_thresholds={best_thresholds}, end up using it={chosen_thresholds is not None}')
    """
    cas_ts = np.array([0.9 for _ in range(cascade_len-1)])
    mname_cas_ts_dict = {}
    # TODO: change to simple evolution strategy because NES is unbounded.
    for i in range(num_trails):
        N = np.random.rand(npop, cascade_len-1)   # samples from a normal distribution N(0,1)
        pop_mnames = []
        model_perf_inftime_df = model_perf_inftime_df[COLS_REPrt[1:]]
        for j in range(npop):
            cas_ts_try = np.clip(cas_ts + noise_std * N[j], a_min=ts_min, a_max=ts_max)
            metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, metric_name, cas_ts_try,
                    problem_type, num_classes, cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
            model_name = f'{cas_starting_str}{model_delim}{i}{model_delim}{j}'
            pop_mnames.append(model_name)
            mname_cas_ts_dict[model_name] = cas_ts_try
            model_perf_inftime_df.loc[model_name] = [metric_value, infer_time]
        model_perf_inftime_df = wrap_rescale_by_pareto_frontier_model(model_perf_inftime_df, val_data, predictor)
        rewards = model_perf_inftime_df.loc[pop_mnames][SCORE].to_numpy()    # (npop,)
        if i % 50 == 0:
            print(f'iter#{i}. {cas_ts=}, rewards={rewards.mean()}')
        # rewards = (rewards - np.mean(rewards)) / np.std(rewards)   # standardize the rewards
        # perform the parameter update.
        cas_ts = cas_ts + lr/(npop*noise_std) * np.dot(N.T, rewards)
        # remove pareto dominated solutions to save space
        pareto_mask = paretoset(model_perf_inftime_df[[PERFORMANCE, SPEED]], sense=['max', 'max'])
        model_perf_inftime_df = model_perf_inftime_df[pareto_mask]
    # the best model/cascade thresholds after NES HPO
    print(model_perf_inftime_df.sort_values(by=SCORE, ascending=False))
    exit(0)
    """
    return chosen_model, chosen_thresholds, time_val, score_val



def get_or_build_partial_weighted_ensemble(predictor: TabularPredictor, base_models: List[str]) -> str:
    global PWE_suffix
    name_suffix = PWE_suffix
    assert len(base_models) > 0
    base_models_set = set(base_models)
    for successor in predictor._trainer.model_graph.successors(base_models[0]):
        precessors = set(predictor._trainer.model_graph.predecessors(successor))
        # cond#1: precessors must be exactlly the same as base_models_set
        # cond#2: have the correct name_suffix
        if precessors == base_models_set and successor.endswith(name_suffix):
            print(f'Retrieve pwe={successor} for {base_models}')
            return successor
    # reach here means no weighted ensemble for current base_models
    successor = predictor.fit_weighted_ensemble(base_models, name_suffix=name_suffix)[0]
    print(f'Fit pwe={successor} for {base_models}')
    return successor


def clean_partial_weighted_ensembles(predictor: TabularPredictor):
    global PWE_suffix
    models_to_delete = []
    for model_name in predictor.get_model_names():
        if model_name.endswith(PWE_suffix):
            models_to_delete.append(model_name)
    predictor.delete_models(models_to_delete=models_to_delete, dry_run=False)


def get_all_predecessor_model_names(predictor: TabularPredictor, model_name: Union[str, List[str]],
        include_self: bool = False) -> Set[str]:
    DAG = predictor._trainer.model_graph
    queue = [model_name] if isinstance(model_name, str) else model_name.copy()
    result = set()
    if include_self:
        result.update(queue)
    while len(queue) > 0:
        cur_model = queue.pop(0)
        for predecessor in DAG.predecessors(cur_model):
            result.add(predecessor)
            queue.append(predecessor)
    return result


def get_non_excuted_predecessors_marginal_time(predictor: TabularPredictor, model_name: str, 
        executed_model_names: Set[str], model_pred_time_dict: Dict[str, float]) -> float:
    """
    Args:
        executed_model_names: this would be updated!!
    """
    predecessor_model_names: Set[str] = get_all_predecessor_model_names(predictor, model_name, include_self=True)
    nonexec_marginal_time_table = 0.0
    for predecessor_model_name in predecessor_model_names:
        if predecessor_model_name in executed_model_names:
            continue
        nonexec_marginal_time_table += model_pred_time_dict[predecessor_model_name]
        executed_model_names.add(predecessor_model_name)
    return nonexec_marginal_time_table


def get_cascade_model_sequence_by_val_marginal_time(predictor: TabularPredictor,
                                                    are_member_of_best: bool = True,
                                                    better_than_prev: bool = True,
                                                    build_pwe_flag: bool = False) -> List[str]:
    """
        Args:
            are_member_of_best: whether to constrain output models must be members of best_model
            better_than_prev: the Pareto heuristic that we only include a model in the cascade if has a higher accuracy than any of the models earlier in the cascade
            build_pwe_flag: whether or not to build partial weighted ensemble
    """
    leaderboard = predictor.leaderboard(silent=True)
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
    if build_pwe_flag:
        predictor.persist_models('all', max_memory=0.75)
        print(f'before build_pwe, {model_sequence=}')
        model_seq_length = len(model_sequence)
        for i in range(model_seq_length-1, 0, -1):
            if model_sequence[i].startswith('WeightedEnsemble'):
                continue
            partial_model_seq = model_sequence[:i+1]
            partial_we_model = get_or_build_partial_weighted_ensemble(predictor, partial_model_seq) 
            model_sequence[i] = partial_we_model
        # predictor.leaderboard()
        predictor.persist_models('all', max_memory=0.75)
        print(f'after build_pwe, {model_sequence=}')
    return model_sequence


def append_approach_exp_result_to_df(exp_result_df: pd.DataFrame, model_name: str, 
        predictor: TabularPredictor, infer_times: float, pred_proba: np.ndarray, 
        test_data: pd.DataFrame, label: str, 
        time_val: Optional[float], score_val: Optional[float]) -> Dict[str, float]:
    meta_cols = get_exp_df_meta_columns(predictor._learner.problem_type)
    speed = test_data.shape[0] / infer_times
    test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
    test_metric1 = test_metrics[meta_cols[MAIN_METRIC_COL]]
    test_metric2 = test_metrics[meta_cols[SEC_METRIC_COL]]
    # the last column is `goodness`, which can only be calculate after getting all models
    exp_result_df.loc[model_name] = [infer_times, speed, test_metric1, test_metric2, time_val, score_val, None]
    return test_metrics


def main(args: argparse.Namespace):
    dataset_name = args.dataset_name
    model_save_path = args.model_save_path
    do_multimodal = args.do_multimodal
    force_training = args.force_training
    exp_result_save_path = args.exp_result_save_path
    presets = args.predictor_presets
    hpo_search_n_trials = args.HPO_random_search_trials
    time_limit = args.time_limit
    ndigits = 4

    for fold_i, n_repeats, train_data, val_data, test_data, label, image_col, eval_metric, model_hyperparameters in load_dataset(dataset_name):
        # TODO: currently only trial on one fold
        if fold_i is not None and fold_i > 0:
            break

        fit_kwargs = dict(
            train_data=train_data,
            tuning_data=val_data,
            hyperparameters=model_hyperparameters,
            presets=presets,
            time_limit=time_limit,
        )
        if do_multimodal:
            # currently support PetFinder or CPP
            # update several fit kwargs
            hyperparameters = get_hyperparameter_config('multimodal')
            if dataset_name == 'PetFinder':
                # Following tutorial, to use first image for each row
                train_data[image_col] = train_data[image_col].apply(lambda ele: ele.split(';')[0])
                test_data[image_col] = test_data[image_col].apply(lambda ele: ele.split(';')[0])
                train_data[image_col] = train_data[image_col].apply(image_id_to_path_petfinder)
                test_data[image_col] = test_data[image_col].apply(image_id_to_path_petfinder)
            elif dataset_name.startswith('CPP-'):
                train_data[image_col] = train_data[image_col].apply(image_id_to_path_cpp)
                test_data[image_col] = test_data[image_col].apply(image_id_to_path_cpp)
            else:
                raise ValueError(f'Currently NOT support do_multimodal with dataset_name={dataset_name}')
            feature_metadata = FeatureMetadata.from_df(train_data)
            feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
            fit_kwargs['hyperparameters'] = hyperparameters
            fit_kwargs['feature_metadata'] = feature_metadata
            fit_kwargs['presets'] = None

        if fold_i is not None:
            model_save_path = f'{model_save_path}/fold{fold_i}'
        if force_training is True or (not os.path.exists(model_save_path)):
            predictor = TabularPredictor(
                label=label,
                eval_metric=eval_metric,
                path=model_save_path,
            )
            predictor.fit(**fit_kwargs)
        else:
            predictor = TabularPredictor.load(model_save_path, require_version_match=False)
        clean_partial_weighted_ensembles(predictor)

        persisted_models = predictor.persist_models('best', max_memory=0.75)
        print(f'persisted_models={persisted_models}')
        # leaderboard = predictor.leaderboard(test_data)   # This is for information displaying
        
        # Load or Create Exp Result df
        if fold_i is not None:
            exp_result_save_path = exp_result_save_path.replace('.csv', f'/fold{fold_i}.csv')
        meta_cols = get_exp_df_meta_columns(predictor._learner.problem_type)
        if os.path.exists(exp_result_save_path):
            exp_result_df = pd.read_csv(exp_result_save_path, index_col='model')
        else:
            exp_result_df = pd.DataFrame(columns=meta_cols).set_index(MODEL).dropna()

        """
        # ==============
        # Infer with each single model of AG stack ensemble
        print('--------')
        best_model = predictor.get_model_best()
        best_model_members = predictor._trainer.get_minimum_model_set(best_model, include_self=False)
        print(f'AG0.4 with best_model={best_model}--{best_model_members}:')
        leaderboard = predictor.leaderboard(silent=True)
        for model_name, can_infer, time_val, score_val in zip(leaderboard['model'], leaderboard['can_infer'], leaderboard['pred_time_val'], leaderboard['score_val']):
            if not can_infer:
                continue
            infer_times = []
            for _ in range(n_repeats):
                ts = time.time()
                pred_proba = predictor.predict_proba(test_data, model=model_name)
                te = time.time()
                infer_times.append(te-ts)
            infer_times = sum(infer_times) / len(infer_times)
            test_metrics = append_approach_exp_result_to_df(exp_result_df, model_name, predictor, infer_times, pred_proba, test_data, label, time_val, score_val)
            if model_name == best_model:
                print(f'AG0.4 best_model {model_name}: {test_metrics} | time: {infer_times}s')
        print('--------')


        # ==============
        # we use Rescale_Pareto (or goodness function) as default
        model_names = ['F2SP/T', 'F2SP++/T', 'F2S/RAND-TM', 'F2S++/RAND-TM', 'F2S/TPE-TM', 'F2S++/TPE-TM']   
        for model_name in model_names:
            print('--------')
            # Set up configs
            if '++' in model_name:
                build_pwe_flag = True
            else:
                build_pwe_flag = False
            if model_name.startswith('F2SP'):
                better_than_prev = True
            else:
                better_than_prev = False
            # Step 1: prepare cascade
            cascade_model_seq = get_cascade_model_sequence_by_val_marginal_time(predictor, better_than_prev=better_than_prev, build_pwe_flag=build_pwe_flag)
            if model_name.endswith('T'):
                chosen_model_name, chosen_threshold, time_val, score_val = hpo_one_param(predictor, eval_metric, 
                        'Rescale_Pareto', False, cascade_model_seq)
            else:
                assert model_name.endswith('TM')
                if model_name.endswith('RAND-TM'):
                    chosen_model_name, chosen_threshold, time_val, score_val \
                            = hpo_multi_params_random_search(predictor, eval_metric, cascade_model_seq, num_trails=hpo_search_n_trials)
                elif model_name.endswith('TPE-TM'):
                    chosen_model_name, chosen_threshold, time_val, score_val \
                            = hpo_multi_params_TPE(predictor, eval_metric, cascade_model_seq, num_trails=hpo_search_n_trials)
                else:
                    raise ValueError(f'not support {model_name=}')
            # Step 2: do inference
            infer_times = []
            for _ in range(n_repeats):
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
            test_metrics = append_approach_exp_result_to_df(exp_result_df, model_name, predictor, infer_times, pred_proba, test_data, label, time_val, score_val)
            print(f'{model_name}: cascade_model_seq={cascade_model_seq}, chosen_threshold={chosen_model_name, chosen_threshold}')
            print(f'{model_name}: score_val={score_val}, time_val={time_val}')
            print(f'{model_name}: {test_metrics} | time: {infer_times}s')
            clean_partial_weighted_ensembles(predictor)
        """

        # store exp_result_df into disk
        # add goodness score col after collecting all model results
        global COLS_REPrt
        model_val_perf_inftime_df = exp_result_df[COLS_REPrt[1:]]
        val_data, _ = helper_get_val_data(predictor)
        model_val_perf_inftime_df = wrap_rescale_by_pareto_frontier_model(model_val_perf_inftime_df, val_data, predictor, mask_pareto_dominated_models=False)
        exp_result_df.update(model_val_perf_inftime_df[[SCORE]])
        exp_result_df = exp_result_df.sort_values(by=[meta_cols[MAIN_METRIC_COL], SPEED], ascending=False)
        print(exp_result_df.drop(columns=SPEED).round(ndigits).reset_index())
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
    parser.add_argument('--predictor_presets', type=str, default='medium_quality')
    parser.add_argument('--HPO_random_search_trials', type=int, default=1000)
    parser.add_argument('--time_limit', type=int, default=None, help='Training time limit in seconds')
    args = parser.parse_args()
    print(f'Exp arguments: {args}')

    main(args)
