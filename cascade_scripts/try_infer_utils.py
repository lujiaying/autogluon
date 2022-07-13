from autogluon.tabular import TabularPredictor, TabularDataset, FeatureMetadata
from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

from .do_no_harm import image_id_to_path_cpp


if __name__ == '__main__':
    # train_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv')
    train_data = TabularDataset('datasets/cpp_research_corpora/2021_60datasets/3564a7a7-0e7c-470f-8f9e-5a029be8e616/train/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet')
    subsample_size = 10000  # subsample subset of data for faster demo, try setting this to much larger values
    if subsample_size is not None and subsample_size < len(train_data):
        train_data = train_data.sample(n=subsample_size, random_state=0)
    train_data.head()
    #label = 'class'
    label = 'label'

    hyperparameters = {
        'XGB': {},
        'GBM': {},
        'CAT': {},
        'RF': {},
        'KNN': {},
        'NN_TORCH': {},
    }
    fit_kwargs = dict(
        train_data=train_data,
        presets='high',
        hyperparameters=hyperparameters,
    )
    image_col = 'image_id'
    hyperparameters = get_hyperparameter_config('multimodal')
    train_data[image_col] = train_data[image_col].apply(image_id_to_path_cpp)
    feature_metadata = FeatureMetadata.from_df(train_data)
    feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    fit_kwargs['hyperparameters'] = hyperparameters
    fit_kwargs['feature_metadata'] = feature_metadata
    fit_kwargs['presets'] = None

    predictor = TabularPredictor(
        label=label,
    )

    predictor.fit(**fit_kwargs)

    #test_data = TabularDataset('https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv')
    test_data = TabularDataset('datasets/cpp_research_corpora/2021_60datasets/3564a7a7-0e7c-470f-8f9e-5a029be8e616/test/part-00001-9c4bc314-0803-4d61-a7c2-6f74f9c9ccfd.c000.snappy.parquet')
    test_data[image_col] = test_data[image_col].apply(image_id_to_path_cpp)

    predictor.persist_models('all')
    leaderboard = predictor.leaderboard(test_data)

    repeats = 2
    infer_dfs = dict()
    for batch_size in [1, 10, 100, 1000, 10000, 100000]:
        infer_df, _ = get_model_true_infer_speed_per_row_batch(data=test_data, predictor=predictor, batch_size=batch_size, repeats=repeats)
        infer_dfs[batch_size] = infer_df
    for key in infer_dfs.keys():
        infer_dfs[key] = infer_dfs[key].reset_index()
        infer_dfs[key]['batch_size'] = key
    import pandas as pd
    infer_df_full = pd.concat([infer_dfs[key] for key in infer_dfs.keys()])
    infer_df_full['rows_per_second'] = 1 / infer_df_full['pred_time_test_with_transform']

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.set_size_inches(12, 12)
    fig.set_dpi(300)

    plt.xscale('log')
    plt.yscale('log')

    models = list(infer_df_full['model'].unique())
    batch_sizes = list(infer_df_full['batch_size'].unique())
    for model in models:
        infer_df_model = infer_df_full[infer_df_full['model'] == model]
        ax.plot(infer_df_model['batch_size'].values, infer_df_model['rows_per_second'].values, label=model)

    ax.set(xlabel='batch_size', ylabel='rows_per_second',
           title='Rows per second inference throughput by data batch_size (CPP-3564a7a7_MM)')
    ax.grid()
    ax.legend()
    fig.savefig('infer_speed_CPP-3564a7a7_MM.png', dpi=300)
    # plt.show()
