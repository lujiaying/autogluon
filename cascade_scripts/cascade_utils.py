"""
Author: Jiaying Lu
Date: May 19
"""

import collections.abc
import copy
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import openml

from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.constants import BINARY, MULTICLASS

# --------------------------------
# Pareto Optimal related functions
# source: https://github.com/tommyod/paretoset/blob/master/paretoset/algorithms_numpy.py
def paretoset(costs, sense=None, distinct=True):
    """Return boolean mask indicating the Pareto set of (non-NaN) numerical data.
    The input data in `costs` can be either a pandas DataFrame or a NumPy ndarray
    of shape (observations, objectives). The user is responsible for dealing with
    NaN values *before* sending data to this function. Only numerical data is
    allowed, with the exception of `diff` (different) columns.
    Parameters
    ----------
    costs : np.ndarray or pd.DataFrame
        Array or DataFrame of shape (observations, objectives).
    sense : list
        List with strings for each column (objective). The value `min` (default)
        indicates minimization, `max` indicates maximization and `diff` indicates
        different values. Using `diff` is equivalent to a group-by operation
        over the columns marked with `diff`. If None, minimization is assumed.
    distinct : bool
        How to treat duplicate rows. If `True`, only the first duplicate is returned.
        If `False`, every identical observation is returned instead.
    Returns
    -------
    mask : np.ndarray
        Boolean mask with `True` for observations in the Pareto set.
    Examples
    --------
    >>> from paretoset import paretoset
    >>> import numpy as np
    >>> costs = np.array([[2, 0], [1, 1], [0, 2], [3, 3]])
    >>> paretoset(costs)
    array([ True,  True,  True, False])
    >>> paretoset(costs, sense=["min", "max"])
    array([False, False,  True,  True])
    The `distinct` parameter:
    >>> paretoset([0, 0], distinct=True)
    array([ True, False])
    >>> paretoset([0, 0], distinct=False)
    array([ True,  True])
    """
    paretoset_algorithm = paretoset_efficient

    costs, sense = validate_inputs(costs=costs, sense=sense)
    assert isinstance(sense, list)

    n_costs, n_objectives = costs.shape

    diff_cols = [i for i in range(n_objectives) if sense[i] == "diff"]
    max_cols = [i for i in range(n_objectives) if sense[i] == "max"]
    min_cols = [i for i in range(n_objectives) if sense[i] == "min"]

    # Check data types (MIN and MAX must be numerical)
    message = "Data must be numerical. Please convert it. Data has type: {}"

    if isinstance(costs, pd.DataFrame):
        data_types = [costs.dtypes.values[i] for i in (max_cols + min_cols)]
        if any(d == np.dtype("O") for d in data_types):
            raise TypeError(message.format(data_types))
    else:
        if costs.dtype == np.dtype("O"):
            raise TypeError(message.format(costs.dtype))

    # No diff columns, use numpy array
    if not diff_cols:
        if isinstance(costs, pd.DataFrame):
            costs = costs.to_numpy(copy=True)
        for col in max_cols:
            costs[:, col] = -costs[:, col]
        return paretoset_algorithm(costs, distinct=distinct)

    n_costs, n_objectives = costs.shape

    # Diff columns are present, use pandas dataframe
    if isinstance(costs, pd.DataFrame):
        df = costs.copy()  # Copy to avoid mutating inputs
        df.columns = np.arange(n_objectives)
    else:
        df = pd.DataFrame(costs)

    assert isinstance(df, pd.DataFrame)
    assert np.all(df.columns == np.arange(n_objectives))

    # If `object` columns are present and they can be converted, do it.
    for col in max_cols:
        df[col] = -pd.to_numeric(df[col], errors="coerce")
    for col in min_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    is_efficient = np.zeros(n_costs, dtype=np.bool_)

    # Create the groupby object
    # We could've implemented our own groupby, but choose to use pandas since
    # it's likely better than what we can come up with on our own.
    groupby = df.groupby(diff_cols)

    # Iteration through the groups
    for key, data in groupby:

        # Get the relevant data for the group and compute the efficient points
        relevant_data = data[max_cols + min_cols].to_numpy(copy=True)
        efficient_mask = paretoset_algorithm(relevant_data.copy(), distinct=distinct)

        # The `pd.DataFrame.groupby.indices` dict holds the row indices of the group
        data_mask = groupby.indices[key]
        is_efficient[data_mask] = efficient_mask

    return is_efficient

def paretoset_efficient(costs, distinct=True):
    """An efficient vectorized algorhtm.

    This algorithm was given by Peter in this answer on Stack Overflow:
    - https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    costs = costs.copy()  # The algorithm mutates the `costs` variable, so we take a copy
    n_costs, n_objectives = costs.shape

    is_efficient = np.arange(n_costs)

    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):

        this_cost = costs[next_point_index]

        # Two points `a` and `b` are *incomparable* if neither dom(a, b) nor dom(b, a).
        # Points that are incomparable to `this_cost`, or dominate `this_cost`.
        # In 2D, these points are below or to the left of `this_cost`.
        current_efficient_points = np.any(costs < this_cost, axis=1)

        # If we're not looking for distinct, keep points equal to this cost
        if not distinct:
            no_smaller = np.logical_not(current_efficient_points)
            equal_to_this_cost = np.all(costs[no_smaller] == this_cost, axis=1)
            current_efficient_points[no_smaller] = np.logical_or(
                current_efficient_points[no_smaller], equal_to_this_cost
            )

        # Any point is incomparable to itself, so keep this point
        current_efficient_points[next_point_index] = True

        # Remove dominated points
        is_efficient = is_efficient[current_efficient_points]
        costs = costs[current_efficient_points]

        # Re-adjust the index
        next_point_index = np.sum(current_efficient_points[:next_point_index]) + 1

    # Create a boolean mask from indices and return it
    is_efficient_mask = np.zeros(n_costs, dtype=np.bool_)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


def validate_inputs(costs, sense=None):
    """Sanitize user inputs for the `paretoset` function.
    Examples
    --------
    >>> costs, sense = validate_inputs([1, 2, 3])
    >>> costs
    array([[1],
           [2],
           [3]])
    """

    # The input is an np.ndarray
    if isinstance(costs, np.ndarray):
        if costs.ndim == 1:
            return validate_inputs(costs.copy().reshape(-1, 1), sense=sense)
        if costs.ndim != 2:
            raise ValueError("`costs` must have shape (observations, objectives).")

        # It's a 2D ndarray -> copy it
        costs = costs.copy()
    elif not isinstance(costs, pd.DataFrame):
            return validate_inputs(np.asarray(costs), sense=sense)
    else:
        return validate_inputs(np.asarray(costs), sense=sense)

    # if (not (isinstance(costs, np.ndarray) and costs.ndim == 2):
    #    raise TypeError("`costs` must be a NumPy array with 2 dimensions or pandas DataFrame.")

    n_costs, n_objectives = costs.shape

    if sense is None:
        return costs, ["min"] * n_objectives
    else:
        sense = list(sense)

    if not isinstance(sense, collections.abc.Sequence):
        raise TypeError("`sense` parameter must be a sequence (e.g. list).")

    if not len(sense) == n_objectives:
        raise ValueError("Length of `sense` must match second dimensions (i.e. columns).")

    # Convert functions "min" and "max" to their names
    sense = [s.__name__ if callable(s) else s for s in sense]
    if not all(isinstance(s, str) for s in sense):
        raise TypeError("`sense` parameter must be a sequence of strings.")
    sense = [s.lower() for s in sense]

    sense_map = {"min": "min", "minimum": "min", "max": "max", "maximum": "max", "diff": "diff", "different": "diff"}

    sense = [sense_map.get(s) for s in sense]

    # Verify that the strings are of correct format
    valid = ["min", "max", "diff"]
    if not all(s in valid for s in sense):
        raise TypeError("`sense` must be one of: {}".format(valid))

    return costs, sense


# --------------------------------
# AG customized score function considering both speed and performance
# Suggested by Nick Erickson

# CONSTANTS
MODEL = 'model'
ERROR = 'error'
ERROR_NORM = 'error_norm'
SCORE = 'goodness'
SPEED = 'speed'
SPEED_ADJUSTED = 'speed_adjusted'
PRED_TIME = 'pred_time_val'
PERFORMANCE = 'score_val'


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
        random_guess_perf: Optional[float] = None, constant: float = 0.05) -> pd.Series:
    if random_guess_perf is None:
        if metric_name == 'roc_auc':
            random_guess_perf = 0.5
        elif metric_name == 'acc':
            random_guess_perf = 0.5
        else:
            raise ValueError(f'apply_extra_penalty_on_error() not support {metric_name=}')
    model_penalty = model_errors / random_guess_perf
    extra_penalty = constant / (model_penalty.clip(lower=1.0) - model_penalty) - constant
    return extra_penalty

def rescale_by_pareto_frontier_model(model_df: pd.DataFrame, speed_soft_cap: float,
        metric_name: str, weights: Tuple[float, float] = (-1.0, 0.01),
        random_guess_perf: Optional[float] = None) -> pd.DataFrame:
    model_df = copy.deepcopy(model_df)
    model_df[ERROR_NORM] = (model_df[ERROR] - model_df[ERROR].min()) / model_df[ERROR].min()
    model_df[ERROR_NORM] = model_df[ERROR_NORM].fillna(0.0)
    model_df[ERROR_NORM] += apply_extra_penalty_on_error(model_df[ERROR].copy(), metric_name, random_guess_perf)
    model_df[SPEED_ADJUSTED] = [adjust_speed(v, soft_cap=speed_soft_cap) for v in model_df[SPEED].values]
    model_df[SPEED_ADJUSTED] = model_df[SPEED_ADJUSTED] / model_df[SPEED_ADJUSTED].min() - 1

    pairs = list(zip(model_df[ERROR_NORM], model_df[SPEED_ADJUSTED]))
    scores = custom_mean(pairs, weights=weights)
    model_df[SCORE] = scores
    return model_df


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


class AGCasGoodness:
    def __init__(self, metric_name: str,
            model_perf_inftime_df: pd.DataFrame, val_data: Tuple[np.ndarray, np.ndarray],
            speed_soft_cap: int = 1000, weights: Tuple[float, float] = (-1.0, 0.01),
            random_guess_perf: Optional[float] = None):
        self.metric_name = metric_name   # indicates type of score_val
        self.model_perf_inftime_df = model_perf_inftime_df
        self.speed_soft_cap = speed_soft_cap
        self.weights = weights
        self._val_nrows = val_data[0].shape[0]
        # Set up random_guess_perf, now support roc_auc and accuracy
        if random_guess_perf is None:
            if metric_name == 'roc_auc':
                random_guess_perf = 0.5
            elif metric_name in ['acc', 'accuracy']:
                random_guess_perf = val_data[1].value_counts(normalize=True).max()
            else:
                raise ValueError(f'Currently NOT support random_guess_perf=`None` for metric={metric_name}')
        assert isinstance(random_guess_perf, float)
        self.random_guess_perf = random_guess_perf
        # Store error_Min and speed_min per models of AG trained stack ensemble
        if ERROR not in model_perf_inftime_df:
            errors = self._cal_error(model_perf_inftime_df[PERFORMANCE])
        else:
            errors = model_perf_inftime_df[ERROR]
        self.error_min = errors.min()
        if SPEED not in model_perf_inftime_df:
            speeds = self._cal_speed(model_perf_inftime_df[PRED_TIME])
        else:
            speeds = model_perf_inftime_df[SPEED]
        self.speed_min = adjust_speed(speeds.min(), speed_soft_cap)

    def _cal_error(self, metric_value: pd.Series) -> pd.Series:
        assert self.metric_name in ['roc_auc', 'acc', 'accuracy']
        return 1.0 - metric_value

    def _cal_speed(self, pred_time: pd.Series) -> pd.Series:
        return self._val_nrows / pred_time

    def __call__(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
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
        model_df[ERROR_NORM] += apply_extra_penalty_on_error(model_df[ERROR], self.metric_name, self.random_guess_perf)
        model_df[SPEED_ADJUSTED] = [adjust_speed(v, soft_cap=self.speed_soft_cap) for v in model_df[SPEED].values]
        model_df[SPEED_ADJUSTED] = model_df[SPEED_ADJUSTED] / model_df[SPEED_ADJUSTED].min() - 1

        pairs = list(zip(model_df[ERROR_NORM], model_df[SPEED_ADJUSTED]))
        scores = custom_mean(pairs, weights=self.weights)
        model_df[SCORE] = scores
        return model_df


class AGCasAccuracy:
    def __init__(self, metric_name: str, infer_time_ubound: float):
        assert metric_name in ['roc_auc', 'acc', 'accuracy']
        self.metric_name = metric_name
        self.const_penlaty = -1e4
        self.infer_time_ubound = infer_time_ubound   # the overall val time upper bound

    def __call__(self, model_perf_inftime_df: pd.DataFrame) -> pd.DataFrame:
        model_perf_inftime_df = model_perf_inftime_df.copy()
        # TODO: may use a smooth version of penalty function
        inftime_penalty = np.where(model_perf_inftime_df[PRED_TIME] <= self.infer_time_ubound, 0.0, self.const_penlaty)
        model_perf_inftime_df[SCORE] = model_perf_inftime_df[PERFORMANCE] + inftime_penalty
        return model_perf_inftime_df


# --------------------------------

# Experiments Reulst DataFrame Related Functions
MAIN_METRIC_COL = 3
SEC_METRIC_COL = 4

def get_exp_df_meta_columns(problem_type: str) -> List[str]:
    # last two columns are metric names
    if problem_type == BINARY:
        meta_cols = ['model', 'pred_time_test', 'speed', 'roc_auc', 'accuracy', 'pred_time_val', 'score_val', 'goodness']
    elif problem_type == MULTICLASS:
        meta_cols = ['model', 'pred_time_test', 'speed', 'accuracy', 'mcc', 'pred_time_val', 'score_val', 'goodness']
    else:
        raise ValueError(f'Invalid input arg problem_type={problem_type}')
    return meta_cols
# --------------------------------

# Load Dataset; some may yeild 10-fold crossvalidation
def load_dataset(dataset_name: str) -> tuple:
    path_val = ''    # by default, dataset not contain validation set
    # Cover Type MultiClass
    if dataset_name == 'CoverTypeMulti':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification/'
        label = 'Cover_Type'
        image_col = None
        path_train = path_prefix + 'train_data.csv'
        path_test = path_prefix + 'test_data.csv'
        eval_metric = 'accuracy'
        model_hyperparameters = 'default'
        n_folds = 1
        n_repeats = 3
    # Adult Income Dataset
    elif dataset_name == 'Inc':
        path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/Inc/'
        label = 'class'
        image_col = None
        path_train = path_prefix + 'train.csv'
        path_test = path_prefix + 'test.csv'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
        n_folds = 1
        n_repeats = 3
    # PetFinder
    elif dataset_name == 'PetFinder':
        path_prefix = 'datasets/petfinder_processed/'
        label = 'AdoptionSpeed'
        image_col = "Images"
        path_train = path_prefix + 'train.csv'
        # We have to use dev, instead of test,
        # since test.csv NOT release labels
        path_test = path_prefix + 'dev.csv'
        eval_metric = 'acc'
        model_hyperparameters = 'default'
        n_folds = 1
        n_repeats = 3
    # CPP one session
    elif dataset_name == 'CPP-6aa99d1a':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/6aa99d1a-1d4b-4d30-bd8b-a26f259b6482/'
        label = 'label'
        image_col = 'image_id'
        path_train = path_prefix + 'train/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-31cb8e7f-4de7-4c5a-8068-d734df5cc6c7.c000.snappy.parquet'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
        n_folds = 1
        n_repeats = 3
    # CPP on session
    elif dataset_name == 'CPP-3564a7a7':
        path_prefix = 'datasets/cpp_research_corpora/2021_60datasets/3564a7a7-0e7c-470f-8f9e-5a029be8e616/'
        label = 'label'
        image_col = 'image_id'
        path_train = path_prefix + 'train/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        path_test = path_prefix + 'test/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet'
        eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
        n_folds = 1
        n_repeats = 3
    elif dataset_name.startswith('openml'):
        """
        # Datasets selected based on being able to
        #  1. train fully on AutoGluon-Medium in <60 seconds on a m5.2xlarge.
        #  2. get significant improvement in score when using AutoGluon-Best.
        datasets = [
            359958,  # 'pc4',
            359947,  # 'MIP-2016-regression',
            190392,  # 'madeline',
            359962,  # 'kc1',
            168911,  # 'jasmine',
            359966,  # 'Internet-Advertisements',
            359954,  # 'eucalyptus',
            168757,  # 'credit-g',
            359950,  # 'boston',
            359956,  # 'qsar-biodeg',
            359975,  # 'Satellite',
            359963,  # 'segment',
            359972,  # 'sylvine',
            359934,  # 'tecator',
            146820,  # 'wilt',
        ]
        """
        task_id = int(dataset_name.split('-')[1])
        task = openml.tasks.get_task(task_id)
        print(task)
        label = task.target_name
        image_col = None
        assert task.task_type_id.name == 'SUPERVISED_CLASSIFICATION'
        if len(task.class_labels) > 2:
            # problem_type = 'multiclass'
            eval_metric = 'accuracy'
        else:
            # problem_type = 'binary'
            eval_metric = 'roc_auc'
        model_hyperparameters = 'default'
        n_repeats, n_folds, n_samples = task.get_split_dimensions()
        assert n_repeats == 1
        assert n_samples == 1
    else:
        print(f'currently not support dataset_name={dataset_name}')
        exit(-1)

    if n_folds == 1:
        # no need to yield folds
        train_data = TabularDataset(path_train)
        val_data = TabularDataset(path_val) if path_val else None
        test_data = TabularDataset(path_test)
        yield None, n_repeats, train_data, val_data, test_data, label, image_col, eval_metric, model_hyperparameters
    else:
        # yeild multiple folds
        # X, y, _, _ = task.get_dataset().get_data(task.target_name)
        all_data, _, _, _ = task.get_dataset().get_data()
        print(f'{dataset_name=} {all_data.shape=}')
        val_data = None
        for fold_idx in range(n_folds):
            train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=fold_idx, sample=0)
            train_data = all_data.loc[train_indices]
            test_data = all_data.loc[test_indices]
            yield fold_idx, n_repeats, train_data, val_data, test_data, label, image_col, eval_metric, model_hyperparameters


if __name__ == '__main__':
    # Test pareto frontier
    # costs = np.array([[1, 1], [0.5, 100], [0.97, 0.87], [1.01, 0.3]])
    # print(f'{costs=}, {paretoset(costs, sense=["max", "max"])}')

    ACC = 'acc'
    speed_soft_caps = [1, 10, 100]  # speed beyond this value is penalized exponentially, contributes less
    weights = (-1, 0.01)  # aka: For every 1% error increase, speed must increase by 100% to compensate

    pairs = [
        ['WeightedEnsemble', 0.9762, 1],  # acc, inf speed
        ['T=0.9', 0.9743, 5.569],  # thresh 0.9
        ['T=0.6', 0.9494, 10.123],  # thresh 0.6
        ['KNN', 0.9694, 72.14],  # KNN
        ['FastAI', 0.92227, 113.51],  # FastAI
        ['RandomGuess', 0.5, 50.0],   # Random guess
        ['BetterThanRand', 0.52, 1.0]
    ]

    model_df = pd.DataFrame(pairs, columns=[MODEL, ACC, SPEED])
    model_df = model_df.set_index(keys=MODEL)
    model_df[ERROR] = 1.0 - model_df[ACC]  # Different for each metric...

    for speed_soft_cap in speed_soft_caps:
        # model_df_out = rescale_by_pareto_frontier_model(model_df=model_df, speed_soft_cap=speed_soft_cap, metric_name=ACC, weights=weights, random_guess_perf=0.5)
        # model_df_out = model_df_out.sort_values(by=[SCORE, ERROR_NORM], ascending=False)
        # test AGCasGoodness class
        fake_val_data = (np.random.rand(1000), None)
        goodness_func = AGCasGoodness(ACC, model_df, fake_val_data, speed_soft_cap, weights, random_guess_perf=0.5)
        # add one new model to simulate the cascade building
        # we may able to find cascade with higher acc and speed
        simulated_pairs = pairs + [['T-TPE', 0.98, 2.5]]
        simulated_df = pd.DataFrame(simulated_pairs, columns=[MODEL, ACC, SPEED]).set_index(MODEL)
        simulated_df[ERROR] = 1.0 - simulated_df[ACC]
        goodness_df_out = goodness_func(simulated_df)
        goodness_df_out = goodness_df_out.sort_values(by=[SCORE, ERROR_NORM], ascending=False)
        print(f'custom_mean (soft_cap={speed_soft_cap}, weights={weights}):')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            # print(model_df_out)
            print('-- to compare diff --')
            print(goodness_df_out)
        print('-----------')
