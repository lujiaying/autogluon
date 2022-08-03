from codecs import namereplace_errors
from enum import Enum
from statistics import mode
from typing import List, Tuple, Dict, Optional, Union, Set, Callable
from functools import partial, reduce
import operator
import itertools
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from scipy.special import softmax
from autogluon.core.constants import BINARY, MULTICLASS
from autogluon.core.metrics import accuracy, roc_auc, log_loss
from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch


# CONSTANTS
MODEL = 'model'
ERROR = 'error'
ERROR_NORM = 'error_norm'
SCORE = 'goodness'
SPEED = 'speed'
SPEED_ADJUSTED = 'speed_adjusted'
PRED_TIME = 'pred_time_val'
PERFORMANCE = 'score_val'
METRIC_FUNC_MAP = {'accuracy': accuracy, 'acc': accuracy, 'roc_auc': roc_auc, 'log_loss': log_loss}
THRESHOLDS_BINARY = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 1.0]
THRESHOLDS_MULTICLASS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
PWE_suffix = '_PWECascade'
COLS_REPrt = [MODEL, PERFORMANCE, PRED_TIME]       # columns for AGCasGoodness
RANDOM_MAGIC_NUM = 0
CASCADE_MNAME = 'Cascade'
DEFAULT_INFER_BATCH_SIZE = 10000
INFER_UTIL_N_REPEATS = 2    


@dataclass(frozen=True)
class CascadeConfig:
    model: Tuple[str]         # cascade model sequence. Length=N
    thresholds: Tuple[float]  # regarding short-circuit/early-exit thresholds. Length=N-1
    pred_time_val: float
    score_val: float
    hpo_score: float
    hpo_func_name: Optional[str] = None


@dataclass(frozen=True)
class F2SP_Preset:
    cascade_algo: str = 'F2S+'
    num_trials: int = 800
    searcher: str = 'TPE'
    hpo_score_func: str = 'ag_goodness'


@dataclass(frozen=True)
class GreedyP_Preset:
    cascade_algo: str = 'Greedy+'
    num_trials: int = 500
    searcher: str = 'TPE'
    hpo_score_func: str = 'ag_goodness'
    each_config_num_trials: int = 50


class HPOScoreFunc(Enum):
    GOODNESS = 'GOODNESS'   # a metric considering both accuracy and speed
    TIME_BOUND_PERFORMANCE = 'TIME_BOUND_PERFORMANCE'   # maximize performance given min speed/infer time limit
    ACCURACY = 'ACCURACY'   # maximize accuracy given min speed/infer time limit; TODO: delete
    # SPEED = 'SPEED'         # maximize speed given min accuracy


class AbstractCasHpoFunc:
    index_constraint: str = MODEL
    columns_constraint: Set[str] = set([PERFORMANCE, PRED_TIME])

    @property
    def name(self):
        return self.name

    def validate_input_df(self, df: pd.DataFrame):
        assert isinstance(df, pd.DataFrame)
        if self.index_constraint != df.index.name:
            raise ValueError(f'AbstractCasHpoFunc Class requires "{self.index_constraint}" as index')
        if not self.columns_constraint.issubset(set(df.columns)):
            raise ValueError(f'AbstractCasHpoFunc Class requires "{self.columns_constraint}" columns contained, but get {df.columns.to_list()}')

    def call(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
        raise ValueError('AbstractCasHpoFunc derived class must define own call() function')

    def __call__(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input_df(model_perf_inftime_df)
        # call must be implemented in derived class
        return self.call(model_perf_inftime_df)


def clean_partial_weighted_ensembles(predictor):
    global PWE_suffix
    models_to_delete = []
    pwe_suffix_full = PWE_suffix + '_FULL'
    for model_name in predictor.get_model_names():
        if model_name.endswith(PWE_suffix) or model_name.endswith(pwe_suffix_full):
            models_to_delete.append(model_name)
    predictor.delete_models(models_to_delete=models_to_delete, dry_run=False)


def helper_get_val_data(predictor) -> Tuple[Tuple[np.ndarray, np.ndarray], bool]:
    """
    For models trained with bagging strategy, 
    we no longer able to directly get val_data
    """
    #TODO: maybe add a upper bound because 25K would be enough for hpo
    val_data = predictor.load_data_internal('val')   # Tuple[np.array, np.array] for X, Y
    is_trained_bagging = False
    if val_data[0] is None and val_data[1] is None:
        val_data = predictor.load_data_internal('train')   # oof_pred actually on train_data
        is_trained_bagging = True
    return val_data, is_trained_bagging


def custom_mean(values, weights):
    scores = []
    for value in values:
        score = 0
        for v, w in zip(value, weights):
            score += v * w
        scores.append(score)
    return scores


def adjust_speed(val, soft_cap, soft_cap_func=np.log10):
    val = val/soft_cap
    if val > 1:
        val = 1 + soft_cap_func(val)
    val = val*soft_cap
    return val


def apply_extra_penalty_on_error(model_errors: pd.Series, metric_name: str,
        random_guess_error: Optional[float] = None, constant: float = 0.05) -> pd.Series:
    if random_guess_error is None:
        if metric_name == 'roc_auc':
            random_guess_error = 0.5
        elif metric_name == 'acc':
            random_guess_error = 0.5
        else:
            raise ValueError(f'apply_extra_penalty_on_error() not support metric_name={metric_name}')
    model_penalty = model_errors / random_guess_error
    extra_penalty = constant / (model_penalty.clip(lower=1.0) - model_penalty) - constant
    return extra_penalty


class AGCasGoodness(AbstractCasHpoFunc):
    name: str = HPOScoreFunc.GOODNESS.name

    def __init__(self, metric_name: str,
            model_perf_inftime_df: pd.DataFrame, val_data: Tuple[np.ndarray, np.ndarray],
            speed_soft_cap: int = 1000, weights: Tuple[float, float] = (-1.0, 0.01),
            random_guess_perf: Optional[float] = None):
        self.metric_name = metric_name   # indicates type of score_val
        self.validate_input_df(model_perf_inftime_df)
        self.model_perf_inftime_df = model_perf_inftime_df
        self.speed_soft_cap = speed_soft_cap
        self.weights = weights
        self._val_nrows = val_data[0].shape[0]
        # Set up random_guess_perf, now support roc_auc and accuracy
        # This is indeed the random guess error
        if random_guess_perf is None:
            if metric_name == 'roc_auc':
                random_guess_perf = 0.5
            elif metric_name in ['acc', 'accuracy']:
                random_guess_perf = val_data[1].value_counts(normalize=True).max()
                # print(f'[DEBUG] random_guess_perf={random_guess_perf} when metric_name={metric_name}')
            elif metric_name in ['log_loss', 'nll']:
                most_freq_class_label = val_data[1].value_counts()[:1].index.item()
                y_pred = np.full(len(val_data[1]), most_freq_class_label)
                random_guess_perf = log_loss(val_data[1], y_pred)
                print(f'[DEBUG] random_guess_perf={random_guess_perf} when metric_name={metric_name}')
            else:
                raise ValueError(f'Currently NOT support random_guess_perf=`None` for metric={metric_name}')
        assert isinstance(random_guess_perf, float)
        self.random_guess_error = self._cal_error(random_guess_perf)
        # Store error_min per model of AG trained stack ensemble
        if ERROR not in model_perf_inftime_df:
            errors = self._cal_error(model_perf_inftime_df[PERFORMANCE])
            self.error_min = errors
        else:
            errors = model_perf_inftime_df[ERROR]
            self.error_min = errors.min()
        # store speed_min per model
        if SPEED not in model_perf_inftime_df:
            speeds = self._cal_speed(model_perf_inftime_df[PRED_TIME])
        else:
            speeds = model_perf_inftime_df[SPEED]
        self.speed_min = adjust_speed(speeds.min(), speed_soft_cap)

    def validate_input_df(self, df: pd.DataFrame):
        # overwrite AbstractCasHpoFunc.validate_input_df()
        assert isinstance(df, pd.DataFrame)
        if self.index_constraint != df.index.name:
            raise ValueError(f'AGCasGoodness Class requires "{self.index_constraint}" as index')
        if not np.isin([self.metric_name, PERFORMANCE, ERROR], df.columns.tolist()).any():
            raise ValueError(f'AGCasGoodness Class requires "{self.metric_name}", "{PERFORMANCE}", or "{ERROR}" columns contained, but get {df.columns.to_list()}')
        if not np.isin([PRED_TIME, SPEED], df.columns.tolist()).any():
            raise ValueError(f'AGCasGoodness Class requires "{PRED_TIME}" or "{SPEED}" columns contained, but get {df.columns.to_list()}')

    def _cal_error(self, metric_value: Union[float, pd.Series]) -> Union[float, pd.Series]:
        assert self.metric_name in ['roc_auc', 'acc', 'accuracy', 'log_loss', 'nll']
        if self.metric_name in ['roc_auc', 'acc', 'accuracy']:
            return 1.0 - metric_value
        elif self.metric_name in ['log_loss', 'nll']:
            return -metric_value

    def _cal_speed(self, pred_time: pd.Series) -> pd.Series:
        return self._val_nrows / pred_time

    def call(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
        # TODO: remove the copy() line to acclerate
        model_df = model_perf_inftime_df.copy()
        # in columns = [MODEL, PERFORMANCE, PRED_TIME]
        # assume we would generate ERROR and SPEED here
        if ERROR not in model_df.columns:
            model_df[ERROR] = self._cal_error(model_df[PERFORMANCE])
        if SPEED not in model_df.columns:
            model_df[SPEED] = self._cal_speed(model_df[PRED_TIME])

        # Main calculation logic
        model_df[ERROR_NORM] = (model_df[ERROR] - self.error_min) / self.error_min
        model_df[ERROR_NORM] = model_df[ERROR_NORM].fillna(0.0)
        model_df[ERROR_NORM] += apply_extra_penalty_on_error(model_df[ERROR], self.metric_name, self.random_guess_error)
        model_df[SPEED_ADJUSTED] = [adjust_speed(v, soft_cap=self.speed_soft_cap) for v in model_df[SPEED].values]
        model_df[SPEED_ADJUSTED] = model_df[SPEED_ADJUSTED] / self.speed_min - 1

        pairs = list(zip(model_df[ERROR_NORM], model_df[SPEED_ADJUSTED]))
        scores = custom_mean(pairs, weights=self.weights)
        model_df[SCORE] = scores
        return model_df


class AGCasAccuracy(AbstractCasHpoFunc):
    name: str = HPOScoreFunc.ACCURACY.name

    def __init__(self, metric_name: str, infer_time_ubound: float):
        assert metric_name in ['roc_auc', 'acc', 'accuracy', 'log_loss', 'nll']
        self.metric_name = metric_name
        self.infer_time_ubound = infer_time_ubound   # the overall val time upper bound
        # the penalty: $f(t) = \alpha * (1 + e^{-(t - \mu) / \beta}) ^ {-1}$
        self.sigmoid_alpha = -2.0   # we will add penalty when time exceeds, assume roc, acc.
        self.sigmoid_mu = infer_time_ubound * 1.1  # when time reach mean, penalty = scale / 2
        # we want $f(\tau) = \alpha / 200 = -0.01$ for \tau is the uboud
        self.sigmoid_beta = - (infer_time_ubound - self.sigmoid_mu) / np.log(199)
        assert self.sigmoid_beta > 0

    def cal_penalty(self, t: pd.Series) -> np.ndarray:
        return self.sigmoid_alpha * expit((t- self.sigmoid_mu) / self.sigmoid_beta)

    def call(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
        model_perf_inftime_df = model_perf_inftime_df.copy()
        # inftime_penalty = np.where(model_perf_inftime_df[PRED_TIME] <= self.infer_time_ubound, 0.0, self.const_penlaty)
        inftime_penalty = np.where(model_perf_inftime_df[PRED_TIME] <= self.infer_time_ubound, 0.0, self.cal_penalty(model_perf_inftime_df[PRED_TIME]))
        model_perf_inftime_df[SCORE] = model_perf_inftime_df[PERFORMANCE] + inftime_penalty
        return model_perf_inftime_df


def get_all_predecessor_model_names(predictor, model_name: Union[str, List[str]],
        include_self: bool = False) -> Set[str]:
    """
    # The leaderboard version is too slow
    leaderboard = predictor.leaderboard(silent=True, extra_info=True)[['model', 'ancestors', 'descendants']]\
            .set_index('model')
    queue = [model_name] if isinstance(model_name, str) else model_name.copy()
    result = set()
    if include_self:
        result.update(queue)
    while len(queue) > 0:
        cur_model = queue.pop(0)
        for predecessor in leaderboard.loc[cur_model]['ancestors']:
            result.add(predecessor)
            queue.append(predecessor)
    print(f'[DEBUG in get_all_predecessor_model_names()] ends {result=} for in {model_name=}')
    """
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


def get_or_build_partial_weighted_ensemble(predictor, base_models: List[str]) -> str:
    global PWE_suffix
    name_suffix = PWE_suffix
    assert len(base_models) > 0
    # base_models_set = set(base_models)
    # for high_quality preset or refit_full models, weighted_ensemble built from ori models
    full_to_ori_dict = predictor.get_model_full_dict(inverse=True)
    refit_full = True if len(full_to_ori_dict) > 0 else False
    base_model_set = set()
    for m in base_models:
        if m in full_to_ori_dict:
            base_model_set.add(full_to_ori_dict[m])
        else:
            base_model_set.add(m)
    for successor in predictor._trainer.model_graph.successors(base_models[0]):
        precessors = set(predictor._trainer.model_graph.predecessors(successor))
        # cond#1: precessors must be exactlly the same as base_models_set
        # cond#2: have the correct name_suffix
        if precessors == base_model_set and successor.endswith(name_suffix):
            # print(f'Retrieve pwe={successor} for {base_models}')
            return successor
    # reach here means no weighted ensemble exists for current base_models
    # TODO: -1 index is risky when fit_weighted_ensemble() changes
    successor = predictor.fit_weighted_ensemble(list(base_model_set), name_suffix=name_suffix, refit_full=refit_full)[-1]
    # print(f'Fit pwe={successor} for {base_models}')
    return successor


def cal_we_pred_proba_and_marginal_time_from_base(predictor,
        model_pred_proba_dict: Dict[str, np.ndarray], 
        model_pred_time_marginal_dict: Dict[str, float],
        we_name: str) -> Tuple[np.ndarray, float]:
    trainer = predictor._trainer
    val_data, is_trained_bagging = helper_get_val_data(predictor)
    # input model_pred_proba_dict, model_pred_time_marginal_dict are changed
    model_pred_proba_dict, model_pred_time_marginal_dict = \
            trainer.get_model_pred_proba_dict(val_data[0], models=[we_name], record_pred_time=True,
                    model_pred_proba_dict=model_pred_proba_dict,
                    model_pred_time_dict=model_pred_time_marginal_dict)
    return model_pred_proba_dict[we_name], model_pred_time_marginal_dict[we_name]


def translate_cascade_sequence_to_WE_version(model_sequence: List[str], predictor,
        model_pred_proba_dict: Optional[Dict[str, np.ndarray]] = None,
        model_pred_time_marginal_dict: Optional[Dict[str, float]] = None,
        ) -> List[str]:
    """
    [Cat, KNN, RF, GBM, WE_L2] -> [Cat, WE2(+KNN), WE3(+RF), WE4(+GBM)==WE_L2]
    """
    last_model = model_sequence[-1]
    assert last_model.startswith('WeightedEnsemble_L')
    if len(model_sequence) <= 2:
        return model_sequence
    predecessors: Set[str] = get_all_predecessor_model_names(predictor, last_model)
    # we assume any model is a member of last WeightedEnsemble
    assert set(model_sequence[:-1]).issubset(predecessors)
    ret_model_sequence = model_sequence.copy()
    for i in range(len(ret_model_sequence)-1, 0, -1):
        if ret_model_sequence[i].startswith('WeightedEnsemble'):
            continue
        partial_model_seq = model_sequence[:i+1]
        if set(partial_model_seq) == predecessors:
            last_model = ret_model_sequence.pop()
            ret_model_sequence[i] = last_model
        else:
            partial_we_model = get_or_build_partial_weighted_ensemble(predictor, partial_model_seq) 
            ret_model_sequence[i] = partial_we_model
            # predictor.persist_models([partial_we_model], with_ancestors=False)
            if model_pred_proba_dict != None and model_pred_time_marginal_dict != None:
                _ = cal_we_pred_proba_and_marginal_time_from_base(predictor, model_pred_proba_dict, model_pred_time_marginal_dict, partial_we_model)
    return ret_model_sequence


def get_cascade_model_sequence_by_val_marginal_time(predictor,
                                                    infer_limit_batch_size: int,
                                                    are_member_of_best: bool = True,
                                                    better_than_prev: bool = True,
                                                    build_pwe_flag: bool = False,
                                                    leaderboard: pd.DataFrame = None,
                                                    ) -> List[str]:
    """
    This carrys Fast-to-Slow and its variants.
    Args:
        are_member_of_best: whether to constrain output models must be members of best_model
        better_than_prev: the Pareto heuristic that we only include a model in the cascade if has a higher accuracy than any of the models earlier in the cascade
        build_pwe_flag: whether or not to build partial weighted ensemble
    """
    # Otherwise conitnue, using model members as candidates
    if leaderboard is None:
        # get genuine infer speed
        global INFER_UTIL_N_REPEATS
        leaderboard = predictor.leaderboard(silent=True)
        n_repeats = INFER_UTIL_N_REPEATS
        # get the genuine pred_time_val_marginal
        val_data, is_trained_bagging = helper_get_val_data(predictor)
        val_label_ori = predictor.transform_labels(val_data[1], inverse=True)
        val_data_wlabel = pd.concat([val_data[0], val_label_ori], axis=1)
        time_per_row_df, _ = get_model_true_infer_speed_per_row_batch(data=val_data_wlabel, predictor=predictor, batch_size=infer_limit_batch_size,
                                                                    repeats=n_repeats, silent=True)
        leaderboard = leaderboard.set_index('model')
        leaderboard.loc[time_per_row_df.index, 'pred_time_val'] = time_per_row_df['pred_time_test'] * len(val_data_wlabel)
        leaderboard.loc[time_per_row_df.index, 'pred_time_val_marginal'] = time_per_row_df['pred_time_test_marginal'] * len(val_data_wlabel)
        leaderboard = leaderboard.reset_index()
    else:
        leaderboard = leaderboard.copy()
    # we have to use last model instead of best model
    best_model_name = predictor.get_model_best()
    if not best_model_name.startswith('WeightedEnsemble_L'):
        # best model NOT contain model member as candidates for cascade sequence
        return [best_model_name]
    # Rule1: from fast to slow
    # Rule2: Layer by Layer
    leaderboard_sorted = leaderboard.sort_values(['stack_level', 'pred_time_val_marginal'], ascending=[True, True])
    model_sequence = leaderboard_sorted['model'].tolist()
    if are_member_of_best:
        valid_cascade_models = predictor._trainer.get_minimum_model_set(best_model_name)
        model_sequence = [m for m in model_sequence if m in valid_cascade_models]
    if build_pwe_flag:
        # print(f'before build_pwe, {model_sequence=}')
        model_sequence = translate_cascade_sequence_to_WE_version(model_sequence, predictor)
        # print(f'after build_pwe, {model_sequence=}')
    leaderboard = predictor.leaderboard(silent=True).set_index(MODEL)
    if better_than_prev:
        #print(f'before build pareto-frontier: {model_sequence=}')
        tmp = []
        max_val_score = 0.0
        for model in model_sequence:
            val_score = leaderboard.loc[model][PERFORMANCE]
            # precessors = set(predictor._trainer.model_graph.predecessors(model))
            # print(f'{model=}, {val_score=}, {precessors=}')
            if val_score <= max_val_score:
                continue
            max_val_score = val_score
            tmp.append(model)
        model_sequence = tmp
        # print(f'after build pareto-frontier: {model_sequence=}')
    return model_sequence


def get_models_pred_proba_on_val(predictor, models: List[str],
        infer_limit_batch_size: int, leaderboard: pd.DataFrame = None,
        ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Assume input leaderboard contains genuine infer speed
    """
    assert isinstance(infer_limit_batch_size, int)
    trainer = predictor._trainer
    # get pred_proba_dict
    val_data, is_trained_bagging = helper_get_val_data(predictor)
    model_pred_proba_dict = {}
    if is_trained_bagging is True:
        # covers bagging strategy
        full_to_ori_mname_dict = predictor.get_model_full_dict(inverse=True)
        as_multiclass: bool = trainer.problem_type == MULTICLASS
        for m in models:
            if m not in full_to_ori_mname_dict:
                model_pred_proba_dict[m] = predictor.get_oof_pred_proba(m, as_multiclass=as_multiclass)
            else:
                # TODO: to check with Nick whether this is a correct simulation
                # high-quality contains refit_full models
                # *_FULL NOT have val_score because all data are used to refit it 
                # so we can only use original model pred_proba as a simulation
                m_ori = full_to_ori_mname_dict[m]
                model_pred_proba_dict[m] = predictor.get_oof_pred_proba(m_ori, as_multiclass=as_multiclass)
    else:
        # models Not use bagging strategy
        model_pred_proba_dict = trainer.get_model_pred_proba_dict(val_data[0], models=models)
    # get genuine infer_time
    val_label_ori = predictor.transform_labels(val_data[1], inverse=True)
    val_data_wlabel = pd.concat([val_data[0], val_label_ori], axis=1)
    if leaderboard is None:
        global INFER_UTIL_N_REPEATS
        n_repeats = INFER_UTIL_N_REPEATS
        time_per_row_df, _ = get_model_true_infer_speed_per_row_batch(data=val_data_wlabel, predictor=predictor, batch_size=infer_limit_batch_size,
                                                                    repeats=n_repeats, silent=True)
        # not including feature transform time
        model_pred_time_marginal_dict = {m: time_per_row_df.loc[m]['pred_time_test_marginal'] * len(val_data_wlabel) for m in models}
    else:
        # !! assume leaderboard contains genuine infer_time
        leaderboard = leaderboard.copy().set_index('model')
        model_pred_time_marginal_dict = {m: leaderboard.loc[m]['pred_time_val_marginal'] * len(val_data_wlabel) for m in models}
    return model_pred_proba_dict, model_pred_time_marginal_dict, val_data


def get_non_excuted_predecessors_marginal_time(predictor, model_name: str, 
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


def get_cascade_metric_and_time_by_threshold(val_data: Tuple[np.ndarray, np.ndarray],
                                             cascade_thresholds: Union[float, List[float], Tuple[float]],
                                             cascade_model_seq: List[str],
                                             model_pred_proba_dict: Dict[str, np.ndarray],
                                             model_pred_time_dict: Dict[str, float],
                                             predictor,
                                             ) -> Tuple[float, float]:
    # mimic logic here: https://github.com/awslabs/autogluon/blob/eb314b1032bc9bc3f611a4d6a0578370c4c89277/core/src/autogluon/core/trainer/abstract_trainer.py#L795
    global METRIC_FUNC_MAP
    metric_name: str = predictor.eval_metric.name
    problem_type = predictor._learner.problem_type
    num_classes = predictor._trainer.num_classes
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
            if threshold == 1.0:
                # TODO: handle non proba case
                # threshold is for probability, th=1.0 means we never early exit at this model
                confident = (pred_proba > threshold)
            else:
                confident = (pred_proba >= threshold) | (pred_proba <= (1-threshold))
        elif problem_type == MULTICLASS:
            if threshold == 1.0:
                # TODO: handle non proba case
                # threshold is for probability, th=1.0 means we never early exit at this model
                confident = (pred_proba > threshold).any(axis=1)
            else:
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


def build_threshold_cands_dynamic(model_pred_proba_dict: Dict[str, np.ndarray],
        problem_type: str,
        quantile_bins: List[float] = [0.2, 0.4, 0.6, 0.8],
        pdf_scale: float = 0.25) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    model_threshold_cands_dict = {}
    for model, pred_proba in model_pred_proba_dict.items():
        if problem_type == BINARY:
            threshold_cands = THRESHOLDS_BINARY.copy()
            proba_for_threshold = np.maximum(pred_proba, 1.0 - pred_proba)
        elif problem_type == MULTICLASS:
            assert len(pred_proba.shape) == 2   # must be 2D shape
            threshold_cands = THRESHOLDS_MULTICLASS.copy()
            proba_for_threshold = np.amax(pred_proba, axis=1)
        else:
            raise ValueError(f'Currently NOT support problem_type={problem_type}')
        quantile = np.quantile(proba_for_threshold, quantile_bins)
        # print(f'{model=}, {quantile=}')
        threshold_cands = np.unique(np.concatenate([quantile, threshold_cands]).round(2))
        threshold_sample_probs = softmax(norm.pdf(threshold_cands, loc=np.median(proba_for_threshold), scale=pdf_scale))
        model_threshold_cands_dict[model] = (threshold_cands, threshold_sample_probs)
    # print(f'{model_threshold_cands_dict=}')
    return model_threshold_cands_dict


def hpo_multi_params_random_search(predictor, cascade_model_seq: List[str],
        hpo_reward_func: AbstractCasHpoFunc,
        infer_limit_batch_size: int,
        num_trails: int = 1000,
        ) -> CascadeConfig:
    """
    Conduct a randommized search over hyperparameters
    Args:
        infer_time_limit: required when hpo_score_func_name=="ACCURACY".
            indicates seconds per row to adhere
    TODO: test whether works for len=1 cascade_model_seq
    """
    rng = np.random.default_rng(RANDOM_MAGIC_NUM)
    problem_type = predictor._learner.problem_type

    # Get val pred proba
    cascade_model_all_predecessors = get_all_predecessor_model_names(predictor, cascade_model_seq, include_self=True)
    model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
            get_models_pred_proba_on_val(predictor, list(cascade_model_all_predecessors), infer_limit_batch_size)
    model_threshold_cands_dict = build_threshold_cands_dynamic(model_pred_proba_dict, problem_type)
    thresholds_cands = []   # (cas_len-1, variable_cand_size)
    thresholds_probs = []
    for model in cascade_model_seq[:-1]:
        thresholds_cands.append(model_threshold_cands_dict[model][0])
        thresholds_probs.append(model_threshold_cands_dict[model][1])
    # Get HPO score using one set of sampled thresholds
    # cascade_perf_inftime_l = []   # model_name, performance, infer_time
    cascade_configs_l: List[CascadeConfig] = []
    if reduce(operator.mul, [len(_) for _ in thresholds_cands], 1) <= num_trails:
        # downgrade to exhaustive search because search space is accountable
        search_space = itertools.product(*thresholds_cands)
    else:
        search_space = []
        for cand, prob in zip(thresholds_cands, thresholds_probs):
            points = rng.choice(cand, size=num_trails, p=prob)
            search_space.append(points)   # leads to (cascade_len-1, num_trails)
        search_space = np.stack(search_space, axis=1)   # (num_trails, cascade_len-1)
    print('[hpo_multi_params_random_search] Start produce val_metrics, and val_time over search space')
    for thresholds in search_space:
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, thresholds,
                cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
        cas_conf = CascadeConfig(model=tuple(cascade_model_seq), thresholds=tuple(thresholds),
                pred_time_val=infer_time, score_val=metric_value, hpo_score=None)  # hpo_score computed later
        cascade_configs_l.append(cas_conf)
        # thresholds_str = cas_delim.join(map(str, thresholds))
        # cascade_perf_inftime_l.append([f'{cas_starting_str}{model_delim}{thresholds_str}', metric_value, infer_time])
    cascade_configs_l = pd.DataFrame(cascade_configs_l).drop_duplicates()
    # made model name unique for hpo_reward_func
    cascade_configs_l[MODEL] = cascade_configs_l[[MODEL, 'thresholds']].apply(tuple, axis=1)
    cascade_configs_l = cascade_configs_l.set_index(MODEL)
    model_perf_inftime_df = hpo_reward_func(cascade_configs_l).sort_values(by=SCORE, ascending=False)
    chosen_row = model_perf_inftime_df.iloc[0]
    chosen_cascd_config = CascadeConfig(model=chosen_row.name[0], thresholds=chosen_row['thresholds'],
            pred_time_val=chosen_row[PRED_TIME], score_val=chosen_row[PERFORMANCE], hpo_score=chosen_row[SCORE],
            hpo_func_name=hpo_reward_func.name)
    return chosen_cascd_config


def hpo_multi_params_TPE(predictor, cascade_model_seq: List[str],
        hpo_reward_func: AbstractCasHpoFunc,
        num_trails: int = 1000, warmup_ntrail_percent = 0.05,
        model_pred_proba_dict: Optional[Dict[str, np.ndarray]] = None,
        model_pred_time_marginal_dict: Optional[Dict[str, float]] = None,
        verbose: bool = True, 
        warmup_cascade_thresholds: List[List[float]] = [],
        is_search_space_continuous: bool = True,
        infer_limit_batch_size: Optional[int] = None,
        ) -> CascadeConfig:
    """
    Use TPE for HP
    Currently use hyperopt implementation

    Args:
        must_return_cascade: do we always return the cascade searched by TPE;
            otherwise, we can return a single model instead.
        model_pred_proba_dict: if not given, invoking trained models to generate.
        model_pred_time_marginal_dict: if not given, invoking trained models to generate.
        verbose: whether to print TPE search process information
        warmup_cascade_thresholds: points added to execute before TPE suggest
        is_search_space_continuous: if not, downgrade to dynamic bins
    """
    def _wrapper_obj_fn(hpo_reward_func: AbstractCasHpoFunc,
            val_data: Tuple[np.ndarray, np.ndarray],
            cascade_model_seq: List[str],
            model_pred_proba_dict: Dict[str, np.ndarray],
            model_pred_time_dict: Dict[str, float],
            predictor,
            cas_ts_try: Tuple[float],
            ) -> float:
        """
        Returns a loss that we want to minimize
        """
        global COLS_REPrt
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, cas_ts_try, 
                cascade_model_seq, model_pred_proba_dict, model_pred_time_dict, predictor)
        cascade_try_name = 'CascadeThreshold_trial'
        model_perf_inftime_df = pd.DataFrame([[cascade_try_name, metric_value, infer_time]], columns=COLS_REPrt).set_index(MODEL)
        model_perf_inftime_df = hpo_reward_func(model_perf_inftime_df)
        reward = model_perf_inftime_df.loc[cascade_try_name][SCORE]
        return -reward

    from hyperopt import fmin, tpe, hp
    rng = np.random.default_rng(RANDOM_MAGIC_NUM)
    cascade_len = len(cascade_model_seq)
    problem_type = predictor._learner.problem_type
    if problem_type == BINARY:
        ts_min = 0.5
        ts_max = 1.0
    elif problem_type == MULTICLASS:
        ts_min = 0.0
        ts_max = 1.0
    else:
        raise ValueError(f'Not support problem_type={problem_type}')

    # Get val pred proba
    val_data = None
    if model_pred_proba_dict == None or model_pred_time_marginal_dict == None:
        cascade_model_all_predecessors = get_all_predecessor_model_names(predictor, cascade_model_seq, include_self=True)
        model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
                get_models_pred_proba_on_val(predictor, list(cascade_model_all_predecessors), infer_limit_batch_size)
    if val_data == None:
        val_data, _ = helper_get_val_data(predictor)
    # no need to proceed to hpo process
    if len(cascade_model_seq) == 1:
        thresholds = (ts_min)
        metric_value, infer_time = get_cascade_metric_and_time_by_threshold(val_data, thresholds,
                cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
        cascade_config = CascadeConfig(
            model=tuple(cascade_model_seq),
            thresholds=(ts_min),
            score_val=metric_value,
            pred_time_val=infer_time,
            hpo_score=None,
            hpo_func_name=None,
        )
        return cascade_config
    # start hpo process
    model_threshold_cands_dict = build_threshold_cands_dynamic(model_pred_proba_dict, problem_type)
    # warmup points define the actual warmup trials
    points_to_warmup = []   # expect size: (warmup_trials, cas_len)
    warmup_ntrials = max(int(warmup_ntrail_percent*num_trails), 10)   # ensure at least try something
    if is_search_space_continuous:
        per_model_warmup_trails_points = []
        for model in cascade_model_seq[:-1]:
            search_points = rng.choice(model_threshold_cands_dict[model][0], 
                    size=warmup_ntrials, 
                    p=model_threshold_cands_dict[model][1])
            per_model_warmup_trails_points.append(search_points)  # final size: (cas_len, warmup_trials)
        for i in range(warmup_ntrials):
            elem = {
                    f't_{j}': per_model_warmup_trails_points[j][i] for j in range(cascade_len-1)
                    }
            points_to_warmup.append(elem)
        # add warmup points from input args
        for thresholds in warmup_cascade_thresholds:
            elem = {f't_{i}': th for i, th in enumerate(thresholds)}
            points_to_warmup.append(elem)
        search_space = [hp.uniform(f't_{i}', ts_min, ts_max) for i in range(cascade_len-1)]
    else:
        # if NOT continuous, use dynamic threshold candidates
        # special logic for points_to_warmup because fmin requires 
        # *indices* instead values as input
        per_model_warmup_trails_points = []
        for model in cascade_model_seq[:-1]:
            search_points = rng.choice(range(len(model_threshold_cands_dict[model][0])), 
                    size=warmup_ntrials, 
                    p=model_threshold_cands_dict[model][1])
            per_model_warmup_trails_points.append(search_points)  # final size: (cas_len, warmup_trials)
        for i in range(warmup_ntrials):
            elem = {
                    f't_{j}': per_model_warmup_trails_points[j][i] for j in range(cascade_len-1)
                    }
            points_to_warmup.append(elem)
        # add warmup points from input args
        for thresholds in warmup_cascade_thresholds:
            elem = {f't_{i}': model_threshold_cands_dict[cascade_model_seq[i]][0].tolist().index(th) for i, th in enumerate(thresholds)}
            points_to_warmup.append(elem)
        search_space = [hp.choice(f't_{i}', model_threshold_cands_dict[m][0].tolist()) for i, m in enumerate(cascade_model_seq[:-1])]
    if verbose:
        print('[hpo_multi_params_TPE] Start produce val_metrics, and val_time over suggested search space')
    # arg `points to evaluate` for fmin as warmup before search_space
    object_func = partial(_wrapper_obj_fn, hpo_reward_func, val_data, cascade_model_seq, 
            model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
    best = fmin(object_func,
        space=search_space,
        algo=tpe.suggest,
        rstate=rng,
        points_to_evaluate=points_to_warmup,
        max_evals=num_trails,
        verbose=verbose,
        show_progressbar=verbose,
        timeout=120)
    best_thresholds = [0.0 for _ in range(cascade_len-1)]
    for k, v in best.items():
        idx = int(k.split('_')[-1])
        if is_search_space_continuous:
            best_thresholds[idx] = v
        else:
            mname = cascade_model_seq[idx]
            v_threshold = model_threshold_cands_dict[mname][0][v]
            best_thresholds[idx] = v_threshold
    score_val, time_val = get_cascade_metric_and_time_by_threshold(val_data, best_thresholds,
            cascade_model_seq, model_pred_proba_dict, model_pred_time_marginal_dict, predictor)
    cascade_perf_inftime_df = pd.DataFrame([[CASCADE_MNAME, score_val, time_val]], columns=COLS_REPrt).set_index(MODEL)
    cascade_perf_inftime_df = hpo_reward_func(cascade_perf_inftime_df)
    cascade_config = CascadeConfig(model=tuple(cascade_model_seq), thresholds=tuple(best_thresholds),
            pred_time_val=time_val, score_val=score_val, hpo_score=cascade_perf_inftime_df.loc[CASCADE_MNAME][SCORE], 
            hpo_func_name=hpo_reward_func.name)
    return cascade_config


def get_model_last(leaderboard: pd.DataFrame) -> str:
    """
    Get last fitted model name
    """
    model_last = leaderboard[leaderboard['fit_order'] == leaderboard['fit_order'].max()]['model'].item()
    return model_last


def get_cascade_model_sequence_by_greedy_search(predictor,
        hpo_reward_func: AbstractCasHpoFunc,
        infer_limit_batch_size: int,
        greedy_search_hpo_ntrials: int = 50,
        build_pwe_flag: bool = False,
        verbose: bool = False,
        leaderboard: pd.DataFrame = None,
        ) -> CascadeConfig:
    """
    First accommodate with AGCasGoodness function.
    Then consider maxamize Accuracy when specifying infer_time_limit
    """
    POS_TO_ADD = 'pos_to_add'   # var for column name
    MODEL_SEQ_ORI = 'model_original'  # var for column name used by `build_pwe_flag`
    problem_type = predictor._learner.problem_type
    if leaderboard is None:
        leaderboard = predictor.leaderboard(silent=True)
        models_to_keep = set(leaderboard[MODEL].tolist())
        models_can_infer = leaderboard[leaderboard['can_infer']][MODEL].tolist()
        model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
                get_models_pred_proba_on_val(predictor, models_can_infer, infer_limit_batch_size)
    else:
        # assume input leaderboard contains genuine infer speed
        models_to_keep = set(leaderboard[MODEL].tolist())
        models_can_infer = leaderboard[leaderboard['can_infer']][MODEL].tolist()
        model_pred_proba_dict, model_pred_time_marginal_dict, val_data = \
                get_models_pred_proba_on_val(predictor, models_can_infer, infer_limit_batch_size,
                leaderboard=leaderboard)
    # WE_final = predictor.get_model_best()
    WE_final = get_model_last(leaderboard)
    assert WE_final.startswith('WeightedEnsemble_L')
    model_cands: Set[str] = get_all_predecessor_model_names(predictor, WE_final)
    assert model_cands.issubset(set(models_can_infer))
    max_cascade_len = len(model_cands) + 1
    # prepare dependencies of models we will use it more than one time
    model_predecessors_dict: Dict[str, Set[str]] = {}
    for model in model_cands:
        model_predecessors = get_all_predecessor_model_names(predictor, model)
        model_predecessors_dict[model] = model_predecessors
    print(f'[INFO] Greedy Search for build cascade sequence WE_final={WE_final}, model_cands={model_cands}')
    # ===== Step 1: greedy selection =====
    patience = 0     # if threshold contains 0.0 or 0.5, patience++
    max_patience = 2
    if problem_type == BINARY:
        patience_min_threshold = 0.5
    elif problem_type == MULTICLASS:
        patience_min_threshold = 0.0
    model_sequence = [WE_final]
    if build_pwe_flag:
        model_sequence_ori = [WE_final]
    config_list: List[CascadeConfig] = []   # stores best config of different length of members
    model_slevel_dict = {m: l for m, l in zip(leaderboard[MODEL], leaderboard['stack_level'])}
    while len(model_cands) > 0:
        msequence_tried_df = []  # store potential sequence for WE_{i+1} if now is WE_{i}
        stack_level_last_pos = {}
        for i, m in enumerate(model_sequence):
            if build_pwe_flag:
                m = model_sequence_ori[i]
            model_slevel = model_slevel_dict[m]
            stack_level_last_pos[model_slevel] = max(i, stack_level_last_pos.get(model_slevel, 0))
        for model in model_cands:
            warmup_cascade_thresholds = []
            # insert model just before WE_final, or right after last model of its layer
            if build_pwe_flag:
                # not using the sequence contains partial_we_model
                # because it is hard to find right position
                msequence_try = model_sequence_ori.copy()
            else:
                msequence_try = model_sequence.copy()
            cur_m_slevel = model_slevel_dict[model]
            if cur_m_slevel in stack_level_last_pos:
                cascade_pos_to_add = stack_level_last_pos[cur_m_slevel] + 1
            else:
                cascade_pos_to_add = len(model_sequence) - 1
            # print(f'[DEBUG] trying to add {model=} at {cascade_pos_to_add=} to {msequence_try=}')
            msequence_try.insert(cascade_pos_to_add, model)
            if len(config_list) > 0:
                prev_best_thresholds = list(config_list[-1].thresholds)
                # prev_best_thresholds NOT contain th for last WE model
                prev_best_thresholds.insert(cascade_pos_to_add, 1.0)
                warmup_cascade_thresholds = [prev_best_thresholds]
            if build_pwe_flag:
                # translate to WE++ version
                msequence_try_ori = msequence_try
                # model_pred_proba_dict, model_pred_time_marginal_dict also stores
                # results for new partial_we_model
                msequence_try = translate_cascade_sequence_to_WE_version(msequence_try_ori, predictor,
                        model_pred_proba_dict, model_pred_time_marginal_dict)
                if len(msequence_try) < len(msequence_try_ori):
                    # WE_n to add is actually WE_final
                    warmup_cascade_thresholds = []
            cascade_config = hpo_multi_params_TPE(predictor, msequence_try, hpo_reward_func, 
                    greedy_search_hpo_ntrials, model_pred_proba_dict=model_pred_proba_dict, 
                    model_pred_time_marginal_dict=model_pred_time_marginal_dict,
                    verbose=verbose, warmup_cascade_thresholds=warmup_cascade_thresholds,
                    infer_limit_batch_size=infer_limit_batch_size,
                    is_search_space_continuous=False)
            # print(f'{chosen_thresholds=} for {msequence_try=}, model specify cascade={chosen_thresholds[cascade_pos_to_add]}')
            tried_dict = {**asdict(cascade_config), **{POS_TO_ADD: cascade_pos_to_add}}
            if build_pwe_flag:
                # store for model candidate selection
                tried_dict[MODEL_SEQ_ORI] = msequence_try_ori
            msequence_tried_df.append(tried_dict)
        msequence_tried_df = pd.DataFrame(msequence_tried_df).sort_values(by='hpo_score', ascending=False)
        # print(msequence_tried_df)
        best_row = msequence_tried_df.iloc[0]
        pos_to_add = best_row[POS_TO_ADD]
        if build_pwe_flag:
            # this is special logic, when last member of WE_final is selected
            #  `pos_to_add` would exceed the length
            model_to_add = best_row[MODEL][pos_to_add] if len(best_row['thresholds']) > pos_to_add else None
            model_to_add_th = best_row['thresholds'][pos_to_add] if len(best_row['thresholds']) > pos_to_add else None
        else:
            model_to_add, model_to_add_th = best_row[MODEL][pos_to_add], best_row['thresholds'][pos_to_add]
        print(f'[INFO] Best insert ({model_to_add}, {model_to_add_th}) gets {best_row["hpo_score"]}. So {model_sequence} --> {best_row[[MODEL, "thresholds"]].to_list()}')
        # early prune model_to_add if regarding threshold is 1.0
        # TODO: add logic if model_to_add_th == 0.5
        if model_to_add_th != 1.0 or len(config_list) == 0:
            # len(config_list) == 0 case, we want to add one config anyway
            model_sequence = list(best_row[MODEL])
            cascade_dict = best_row.to_dict()
            cascade_dict.pop(POS_TO_ADD)
            if build_pwe_flag:
                model_sequence_ori = list(best_row[MODEL_SEQ_ORI])
                cascade_dict.pop(MODEL_SEQ_ORI)
            cascade_config = CascadeConfig(**cascade_dict)
            config_list.append(cascade_config)
            # after add one config, early prune
            if model_to_add_th == 1.0:
                break
            # only try max_patience times of definitelly exit on certain member
            if patience_min_threshold in cascade_config.thresholds:
                patience += 1
                if patience >= max_patience:
                    break
        else:
            # model_to_add with threshold=1.0 means no gain after adding more exit points
            # do early prune
            break
        # reach here means going to find next best candidate to add
        if build_pwe_flag:
            model_to_add_ori = best_row[MODEL_SEQ_ORI][pos_to_add]
            print(f'[INFO] {model_to_add} is built on {best_row[MODEL_SEQ_ORI][:pos_to_add+1]}')
            model_cands.remove(model_to_add_ori)
        else:
            model_cands.remove(model_to_add)
    # ===== Step 2: greedy pruning =====
    print('Finish the greedy adding process, now proceed to greedy pruning process from ...')
    if len(config_list) > 0:
        best_config = config_list[-1]
        model_sequence = list(best_config.model)
        best_hpo_score = best_config.hpo_score
        print(f'best cascade config after selection: {best_config}')
    else:
        model_sequence = []
        best_hpo_score = None
    # at least contains 2 member for pruning
    while len(model_sequence) > 2:
        msequence_tried_df = []  # store potential sequence for WE_{i+1} if now is WE_{i}
        for midx, model in enumerate(model_sequence[:-1]):
            msequence_try = model_sequence.copy()
            msequence_try.pop(midx)
            prev_best_thresholds = list(config_list[-1].thresholds)
            prev_best_thresholds.pop(midx)
            warmup_cascade_thresholds = [prev_best_thresholds]
            cascade_config = hpo_multi_params_TPE(predictor, msequence_try, hpo_reward_func, 
                            greedy_search_hpo_ntrials, model_pred_proba_dict=model_pred_proba_dict, 
                            model_pred_time_marginal_dict=model_pred_time_marginal_dict,
                            verbose=verbose, warmup_cascade_thresholds=warmup_cascade_thresholds,
                            infer_limit_batch_size=infer_limit_batch_size,
                            is_search_space_continuous=False)
            tried_dict = {**asdict(cascade_config), **{POS_TO_ADD: midx}}
            msequence_tried_df.append(tried_dict)
        msequence_tried_df = pd.DataFrame(msequence_tried_df).sort_values(by='hpo_score', ascending=False)
        # print(msequence_tried_df)
        best_row = msequence_tried_df.iloc[0]
        pos_to_prune = best_row[POS_TO_ADD]
        model_to_prune = best_row[MODEL][pos_to_prune]
        cur_hpo_score = best_row["hpo_score"]
        print(f'[INFO] remove {model_to_prune} from {config_list[-1].model} gets {cur_hpo_score} -> {best_row[[MODEL, "thresholds"]].to_list()}')
        if cur_hpo_score >= best_hpo_score:
            model_sequence = list(best_row[MODEL])
            cascade_dict = best_row.to_dict()
            cascade_dict.pop(POS_TO_ADD)
            cascade_config = CascadeConfig(**cascade_dict)
            config_list.append(cascade_config)
            best_hpo_score = cascade_config.hpo_score
        else:
            break
    # ===== Step 3: try continuous threshold search on top2 cascade sequence =====
    saved_confs_len = len(config_list)
    topk = 2
    config_list_df = pd.DataFrame(config_list).dropna().sort_values(by='hpo_score', ascending=False)
    best_config = CascadeConfig(**config_list_df.iloc[0].to_dict())
    for row_idx in range(min(topk, saved_confs_len)):
        row = config_list_df.iloc[row_idx]
        model_sequence = list(row.model)
        thresholds = list(row.thresholds)
        warmup_cascade_thresholds = [thresholds]
        cascade_config = hpo_multi_params_TPE(predictor, model_sequence, hpo_reward_func, 
                        2*greedy_search_hpo_ntrials, model_pred_proba_dict=model_pred_proba_dict, 
                        model_pred_time_marginal_dict=model_pred_time_marginal_dict,
                        verbose=verbose, warmup_cascade_thresholds=warmup_cascade_thresholds)
        if cascade_config.hpo_score >= best_config.hpo_score:
            best_config = cascade_config
    # clean up unused partial_we_models
    if build_pwe_flag:
        full_to_ori_dict = predictor.get_model_full_dict(inverse=True)
        for m in best_config.model:
            models_to_keep.add(m)
            if m.endswith('_FULL'):
                models_to_keep.add(full_to_ori_dict[m])
        predictor.delete_models(models_to_keep=list(models_to_keep), dry_run=False)
    return best_config


def prune_cascade_config(chosen_cascd_config: CascadeConfig, problem_type: str):
    # TODO: if searched thresholds contains 0.5 or 1.0, we can prune it
    if problem_type == BINARY:
        min_threshold = 0.5
    elif problem_type == MULTICLASS:
        min_threshold = 0.0
    else:
        raise AssertionError(f'Invalid cascade problem_type: {problem_type}')
    defin_exit_idx = None
    for idx, threshold in enumerate(chosen_cascd_config.thresholds):
        if threshold == min_threshold:
            # at `idx`, cascade will definitely early exit
            defin_exit_idx = idx
    if defin_exit_idx is not None:
        if defin_exit_idx == 0:
           cascd_config = CascadeConfig(model=tuple([chosen_cascd_config.model[0]]), thresholds=tuple([min_threshold]),
                    pred_time_val=chosen_cascd_config.pred_time_val, score_val=chosen_cascd_config.score_val, 
                    hpo_score=chosen_cascd_config.hpo_score, hpo_func_name=chosen_cascd_config.hpo_func_name
                    ) 
        else:
           cascd_config = CascadeConfig(model=tuple(chosen_cascd_config.model[:defin_exit_idx+1]), 
                    thresholds=tuple(chosen_cascd_config.thresholds[:defin_exit_idx]),
                    pred_time_val=chosen_cascd_config.pred_time_val, score_val=chosen_cascd_config.score_val, 
                    hpo_score=chosen_cascd_config.hpo_score, hpo_func_name=chosen_cascd_config.hpo_func_name
                    ) 
        print(f'before {chosen_cascd_config}, after {cascd_config}')
        return cascd_config
    return chosen_cascd_config


def hpo_post_process(predictor, chosen_cascd_config: CascadeConfig,
        hpo_reward_func: AbstractCasHpoFunc, leaderboard: pd.DataFrame) -> CascadeConfig:
    """
    examine whether single model is better than the chosen cascade_config.
    If so, return a cascade_config with one single model
    Here we still only access to validation data set
    """
    global COLS_REPrt, CASCADE_MNAME
    model_perf_inftime_df = leaderboard[leaderboard['can_infer']][COLS_REPrt].copy().set_index(MODEL)
    model_perf_inftime_df.loc[CASCADE_MNAME] = [chosen_cascd_config.score_val, chosen_cascd_config.pred_time_val]
    model_perf_inftime_df = hpo_reward_func(model_perf_inftime_df).sort_values(by=SCORE, ascending=False)
    best_row = model_perf_inftime_df.iloc[0]
    if best_row.name == CASCADE_MNAME:
        problem_type = predictor._learner.problem_type
        return prune_cascade_config(chosen_cascd_config, problem_type)
    else:
        single_mem_cascd_config = CascadeConfig(model=tuple([best_row.name]), thresholds=tuple([0.0]),
                    pred_time_val=best_row[PRED_TIME], score_val=best_row[PERFORMANCE], hpo_score=best_row[SCORE],
                    )
        return single_mem_cascd_config