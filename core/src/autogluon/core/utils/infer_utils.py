import copy
import time
import numpy as np
import pandas as pd


def get_model_true_infer_speed_per_row_batch(
        data,
        *,
        predictor,
        batch_size: int = 100000,
        repeats=1,
        silent=False):
    """
    Get per-model true inference speed per row for a given batch size of data.

    Parameters
    ----------
    data : :class:`TabularDataset` or :class:`pd.DataFrame`
        Table of the data, which is similar to a pandas DataFrame.
        Must contain the label column to be compatible with leaderboard call.
    predictor : TabularPredictor
        Fitted predictor to get inference speeds for.
    batch_size : int, default = 100000
        Batch size to use when calculating speed. `data` will be modified to have this many rows.
        If simulating large-scale batch inference, values of 100000+ are recommended to get genuine throughput estimates.
    repeats : int, default = 1
        Repeats of calling leaderboard. Repeat times are averaged to get more stable inference speed estimates.
    silent : False
        If False, logs information regarding the speed of each model + feature preprocessing.

    Returns
    -------
    time_per_row_df : pd.DataFrame, time_per_row_transform : float
        time_per_row_df contains each model as index.
            'pred_time_test_with_transform' is the end-to-end prediction time per row in seconds if calling `predictor.predict(data, model=model)`
            'pred_time_test' is the end-to-end prediction time per row in seconds minus the global feature preprocessing time.
            'pred_time_test_marginal' is the prediction time needed to predict for this particular model minus dependent model inference times and global preprocessing time.
        time_per_row_transform is the time in seconds per row to do the feature preprocessing.
    """
    data_batch = copy.deepcopy(data)
    len_data = len(data_batch)
    if len_data == batch_size:
        pass
    elif len_data < batch_size:
        # add more rows
        duplicate_count = int(np.ceil(batch_size / len_data))
        data_batch = pd.concat([data_batch for _ in range(duplicate_count)])
        len_data = len(data_batch)
    if len_data > batch_size:
        # sample rows
        data_batch = data_batch.sample(n=batch_size, random_state=0)
        len_data = len(data_batch)

    if len_data != batch_size:
        raise AssertionError(f'len(data_batch) must equal batch_size! ({len_data} != {batch_size})')

    ts = time.time()
    for i in range(repeats):
        predictor.transform_features(data_batch)
    time_transform = (time.time() - ts) / repeats

    # delete these models to speed up, due to some reason these bag models 'can_infer' == True
    models_to_delete = set(["ImagePredictor_BAG_L1", "TextPredictor_BAG_L1"])
    can_infer_models = set(predictor.get_model_names(can_infer=True))
    models_to_delete = list(models_to_delete.intersection(can_infer_models))
    if len(models_to_delete) > 0:
         import tempfile
         print(f'[DEBUG] delete models: for get_genuine_speed {models_to_delete}')
         with tempfile.TemporaryDirectory() as tmp_dir:
            # will clean up tmp dir afterwards
            predictor_cp = predictor.clone(tmp_dir, require_version_match=False, return_clone=True, dirs_exist_ok=True)
            predictor_cp.delete_models(models_to_delete=models_to_delete, delete_from_disk=False, 
                                        allow_delete_cascade=True, dry_run=False)
            predictor_cp.persist_models(models='all', max_memory=0.5)
    else:
        predictor_cp = predictor
        predictor_cp.persist_models(models='all', max_memory=0.5)
    leaderboards = []
    for i in range(repeats):
        leaderboard = predictor_cp.leaderboard(data_batch, silent=True)
        leaderboard = leaderboard[leaderboard['can_infer']][['model', 'pred_time_test', 'pred_time_test_marginal']]
        leaderboard = leaderboard.set_index('model')
        leaderboards.append(leaderboard)
    leaderboard = pd.concat(leaderboards)
    time_per_batch_df = leaderboard.groupby(level=0).mean()
    time_per_batch_df['pred_time_test_with_transform'] = time_per_batch_df['pred_time_test'] + time_transform
    time_per_row_df = time_per_batch_df / batch_size
    time_per_row_transform = time_transform / batch_size

    if not silent:
        for index, row in time_per_row_df.iterrows():
            time_per_row = row['pred_time_test_with_transform']
            time_per_row_print = time_per_row
            unit = 's'
            if time_per_row_print < 1e-2:
                time_per_row_print *= 1000
                unit = 'ms'
                if time_per_row_print < 1e-2:
                    time_per_row_print *= 1000
                    unit = 'μs'
            print(f"{round(time_per_row_print, 3)}{unit} per row | {index}")
        time_per_row_transform_print = time_per_row_transform
        unit = 's'
        if time_per_row_transform_print < 1e-2:
            time_per_row_transform_print *= 1000
            unit = 'ms'
            if time_per_row_transform_print < 1e-2:
                time_per_row_transform_print *= 1000
                unit = 'μs'
        print(f"{round(time_per_row_transform_print, 3)}{unit} per row | transform_features")

    predictor_cp.unpersist_models()
    return time_per_row_df, time_per_row_transform