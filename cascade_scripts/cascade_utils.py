"""
Author: Jiaying Lu
Date: May 19
"""

import collections.abc
import copy
from typing import Tuple, Optional, Callable, List

import numpy as np
import pandas as pd

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

MODEL = 'model'
ERROR = 'error'
ERROR_NORM = 'error_norm'
SCORE = 'goodness'
SPEED = 'speed'
SPEED_ADJUSTED = 'speed_adjusted'


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

if __name__ == '__main__':
    # Test pareto frontier
    # costs = np.array([[1, 1], [0.5, 100], [0.97, 0.87], [1.01, 0.3]])
    # print(f'{costs=}, {paretoset(costs, sense=["max", "max"])}')

    ACC = 'acc'
    speed_soft_caps = [1, 10, 100]  # speed beyond this value is penalized exponentially, contributes less
    weights = [-1, 0.01]  # aka: For every 1% error increase, speed must increase by 100% to compensate

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
    model_df = model_df.set_index(keys=MODEL, drop=True)
    model_df[ERROR] = 1.0 - model_df[ACC]  # Different for each metric...

    for speed_soft_cap in speed_soft_caps:
        model_df_out = rescale_by_pareto_frontier_model(model_df=model_df, speed_soft_cap=speed_soft_cap, metric_name=ACC, weights=weights, random_guess_perf=0.5)
        model_df_out = model_df_out.sort_values(by=[SCORE, ERROR_NORM], ascending=False)
        print(f'custom_mean (soft_cap={speed_soft_cap}, weights={weights}):')
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(model_df_out)
        print('-----------')
