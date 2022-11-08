import copy
import os
import shutil
import uuid
import pytest
import time
from typing import List, Union
import numpy as np

from autogluon.core.utils import download, unzip
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerMulticlassToBinary
from autogluon.core.utils import infer_problem_type, generate_train_test_split
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular import TabularDataset, TabularPredictor


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runregression", action="store_true", default=False, help="run regression tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "regression: mark test as regression test")


def pytest_collection_modifyitems(config, items):
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_regression = pytest.mark.skip(reason="need --runregression option to run")
    custom_markers = dict(
        slow=skip_slow,
        regression=skip_regression
    )
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        custom_markers.pop("slow", None)
    if config.getoption("--runregression"):
        # --runregression given in cli: do not skip slow tests
        custom_markers.pop("regression", None)

    for item in items:
        for marker in custom_markers:
            if marker in item.keywords:
                item.add_marker(custom_markers[marker])


class DatasetLoaderHelper:
    dataset_info_dict = dict(
        # Binary dataset
        adult={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/AdultIncomeBinaryClassification.zip',
            'name': 'AdultIncomeBinaryClassification',
            'problem_type': BINARY,
            'label': 'class',
        },
        # Multiclass big dataset with 7 classes, all features are numeric. Runs SLOW.
        covertype={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification.zip',
            'name': 'CoverTypeMulticlassClassification',
            'problem_type': MULTICLASS,
            'label': 'Cover_Type',
        },
        # Subset of covertype dataset with 3k train/test rows. Ratio of labels is preserved.
        covertype_small={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassificationSmall.zip',
            'name': 'CoverTypeMulticlassClassificationSmall',
            'problem_type': MULTICLASS,
            'label': 'Cover_Type',
        },
        # Regression with mixed feature-types, skewed Y-values.
        ames={
            'url': 'https://autogluon.s3.amazonaws.com/datasets/AmesHousingPriceRegression.zip',
            'name': 'AmesHousingPriceRegression',
            'problem_type': REGRESSION,
            'label': 'SalePrice',
        },
        # Regression with multiple text field and categorical
        sts={
            'url': 'https://autogluon-text.s3.amazonaws.com/glue_sts.zip',
            'name': 'glue_sts',
            'problem_type': REGRESSION,
            'label': 'score',
        }
    )

    @staticmethod
    def load_dataset(name: str, directory_prefix: str = './datasets/'):
        dataset_info = copy.deepcopy(DatasetLoaderHelper.dataset_info_dict[name])
        train_file = dataset_info.pop('train_file', 'train_data.csv')
        test_file = dataset_info.pop('test_file', 'test_data.csv')
        name_inner = dataset_info.pop('name')
        url = dataset_info.pop('url', None)
        train_data, test_data = DatasetLoaderHelper.load_data(
            directory_prefix=directory_prefix,
            train_file=train_file,
            test_file=test_file,
            name=name_inner,
            url=url,
        )

        return train_data, test_data, dataset_info

    @staticmethod
    def load_data(directory_prefix, train_file, test_file, name, url=None):
        if not os.path.exists(directory_prefix):
            os.mkdir(directory_prefix)
        directory = directory_prefix + name + "/"
        train_file_path = directory + train_file
        test_file_path = directory + test_file
        if (not os.path.exists(train_file_path)) or (not os.path.exists(test_file_path)):
            # fetch files from s3:
            print("%s data not found locally, so fetching from %s" % (name, url))
            zip_name = download(url, directory_prefix)
            unzip(zip_name, directory_prefix)
            os.remove(zip_name)

        train_data = TabularDataset(train_file_path)
        test_data = TabularDataset(test_file_path)
        return train_data, test_data


class FitHelper:
    @staticmethod
    def fit_and_validate_dataset(dataset_name, fit_args, sample_size=1000, refit_full=True, model_count=1, delete_directory=True, extra_metrics=None):
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label = dataset_info['label']
        save_path = os.path.join(directory_prefix, dataset_name, f'AutogluonOutput_{uuid.uuid4()}')
        init_args = dict(
            label=label,
            path=save_path,
        )
        predictor = FitHelper.fit_dataset(train_data=train_data, init_args=init_args, fit_args=fit_args, sample_size=sample_size)
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)
        predictor.predict(test_data)
        pred_proba = predictor.predict_proba(test_data)
        predictor.evaluate(test_data)
        predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba)

        model_names = predictor.get_model_names()
        model_name = model_names[0]
        assert len(model_names) == (model_count + 1)
        if refit_full:
            refit_model_names = predictor.refit_full()
            assert len(refit_model_names) == (model_count + 1)
            refit_model_name = refit_model_names[model_name]
            assert '_FULL' in refit_model_name
            predictor.predict(test_data, model=refit_model_name)
            predictor.predict_proba(test_data, model=refit_model_name)
        predictor.info()
        predictor.leaderboard(test_data, extra_info=True, extra_metrics=extra_metrics)

        assert os.path.realpath(save_path) == os.path.realpath(predictor.path)
        if delete_directory:
            shutil.rmtree(save_path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.
        return predictor

    @staticmethod
    def fit_and_validate_dataset_with_cascade(dataset_name, fit_args, cascade: List[str], sample_size=1000, refit_full=True, model_count=1, delete_directory=True):
        predictor = FitHelper.fit_and_validate_dataset(
            dataset_name=dataset_name,
            fit_args=fit_args,
            sample_size=sample_size,
            refit_full=refit_full,
            model_count=model_count,
            delete_directory=False,
        )
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)

        predictor.predict(test_data, model=cascade)
        predictor.predict_proba(test_data, model=cascade)

        if delete_directory:
            shutil.rmtree(predictor.path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.
        return predictor

    @staticmethod
    def fit_dataset(train_data, init_args, fit_args, sample_size=None):
        if sample_size is not None and sample_size < len(train_data):
            train_data = train_data.sample(n=sample_size, random_state=0)
        return TabularPredictor(**init_args).fit(train_data, **fit_args)

    @staticmethod
    def check_cascade_speed_accuracy_simulation(dataset_name: str, fit_args: dict, cascade: List[str], cascade_thresholds: List[float],
                                                train_sample_size: int = 1000, infer_limit_batch_size: int = 100000, 
                                                n_repeats: int = 10, simulation_time_diff: float = 0.1, 
                                                delete_directory: bool = True):
        from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch
        from autogluon.tabular.predictor.cascade_do_no_harm import get_cascade_metric_and_time_by_threshold, get_all_predecessor_model_names
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label = dataset_info['label']
        save_path = os.path.join(directory_prefix, dataset_name, f'AutogluonOutput_{uuid.uuid4()}')
        init_args = dict(
            label=label,
            path=save_path,
        )
        predictor = FitHelper.fit_dataset(train_data, init_args, fit_args, sample_size=train_sample_size)
        predictor.persist_models('all')
        predictor.leaderboard()
        cascade_last_model = cascade[-1]
        print(f'[INFO] constitutent of {cascade_last_model}: {get_all_predecessor_model_names(predictor, cascade_last_model)}')
        test_data_sampled = test_data.sample(n=infer_limit_batch_size, replace=True, random_state=0)
        # Get actual end2end infer time and accuracy
        trainer = predictor._learner.load_trainer()
        actual_infer_times = []
        actual_feat_trans_times = []
        for i in range(n_repeats):
            ts = time.time()
            test_data_sampled_X = predictor.transform_features(test_data_sampled)
            te_transform = time.time()
            _ = trainer.get_model_pred_proba_dict(
                test_data_sampled_X, cascade, fit=False,
                cascade=True, cascade_threshold=cascade_thresholds,
                )
            te = time.time()
            actual_infer_times.append(te - ts)
            actual_feat_trans_times.append(te_transform - ts)
        actual_feat_trans_time_prow = np.mean(actual_feat_trans_times) / infer_limit_batch_size
        actual_infer_time_prow = np.mean(actual_infer_times) / infer_limit_batch_size
        test_data_X = predictor.transform_features(test_data)
        test_data_y = predictor.transform_labels(test_data[label])
        model_pred_proba_dict = trainer.get_model_pred_proba_dict(
            test_data_X, cascade, fit=False,
            cascade=True, cascade_threshold=cascade_thresholds,
            )
        actual_pred_proba = model_pred_proba_dict[cascade[-1]]
        if predictor._learner.problem_type == BINARY:
            actual_pred_proba = LabelCleanerMulticlassToBinary.convert_binary_proba_to_multiclass_proba(actual_pred_proba)
        #print(f'[DEBUG] {test_data[label]=}; {test_data[label].unique()=}; {actual_pred_proba=} {actual_pred_proba.shape=}')
        actual_eval_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=actual_pred_proba, silent=True)
        # Get simulated infer time and accuracy
        time_per_row_df, time_per_row_transform = get_model_true_infer_speed_per_row_batch(data=test_data_sampled, 
                                                    predictor=predictor, batch_size=infer_limit_batch_size,
                                                    repeats=n_repeats)
        print(f'[DEBUG] time_per_row_df={time_per_row_df}')
        simu_model_pred_proba_dict = {}
        simu_model_pred_time_dict = {}   # stores marginal time w/o feat trans predcting all test_data
        for index, row in time_per_row_df.iterrows():
            simu_model_pred_time_dict[index] = row['pred_time_test_marginal'] * len(test_data)
            pred_proba = predictor.predict_proba(data=test_data, model=index, as_multiclass=(predictor._learner.problem_type == MULTICLASS))
            simu_model_pred_proba_dict[index] = pred_proba
        simu_score, simu_time_no_trans = \
            get_cascade_metric_and_time_by_threshold((test_data_X, test_data_y), cascade_thresholds, cascade,
                                                    simu_model_pred_proba_dict, simu_model_pred_time_dict, predictor)
        simu_time_no_trans_prow = simu_time_no_trans / len(test_data)
        simu_time_prow = simu_time_no_trans_prow + time_per_row_transform
        # check whether simulation aligns with reality
        print(f'ACTUAL infer sec/row={actual_infer_time_prow}, feat trans sec/row={actual_feat_trans_time_prow}, infer_batch_size={infer_limit_batch_size}')
        print(f'ACTUAL eval metrics: {actual_eval_metrics}')
        print(f'SIMULATION infer_time sec/row={simu_time_prow}, feat trans sec/row={time_per_row_transform}, {predictor.eval_metric.name}={simu_score}')
        func_check_closeness = lambda x, y: abs(x - y) <= simulation_time_diff * max(x, y)
        assert simu_score == actual_eval_metrics[predictor.eval_metric.name]
        assert func_check_closeness(time_per_row_transform, actual_feat_trans_time_prow)
        assert func_check_closeness(simu_time_prow, actual_infer_time_prow)
        # clean up everything
        if delete_directory is True:
            shutil.rmtree(predictor.path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.

    @staticmethod
    def check_fit_cascade_infer_limit(dataset_name: str, fit_args: dict, 
                                      infer_limit: float, infer_limit_batch_size: int, 
                                      hyperparameter_cascade: Union[str, dict],
                                      train_sample_size: int = 1000, 
                                      n_repeats: int = 10, cascade_time_diff: float = 0.2, 
                                      delete_directory: bool = True):
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label = dataset_info['label']
        save_path = os.path.join(directory_prefix, dataset_name, f'AutogluonOutput_{uuid.uuid4()}')
        init_args = dict(
            label=label,
            path=save_path,
        )
        predictor = FitHelper.fit_dataset(train_data, init_args, fit_args, sample_size=train_sample_size)
        predictor.persist_models('all')
        predictor.leaderboard()
        # fit cascade
        chosen_cascade_conf = predictor.fit_cascade(train_data, infer_limit, infer_limit_batch_size, hyperparameter_cascade=hyperparameter_cascade)
        print(f'[INFO] Get return cascade config={chosen_cascade_conf}')
        # do infer
        ## get genuine infer speed with infer_limit_batch_size
        test_data_sampled = test_data.sample(n=infer_limit_batch_size, replace=True, random_state=0)
        pred_time_test_list = []
        for i in range(n_repeats):
            pred_time_test, _ = predictor.do_infer_with_cascade_conf(chosen_cascade_conf, test_data_sampled)
            pred_time_test_list.append(pred_time_test)
        pred_time_test = np.mean(pred_time_test_list)
        pred_time_test_per_row = pred_time_test / infer_limit_batch_size
        _, pred_proba = predictor.do_infer_with_cascade_conf(chosen_cascade_conf, test_data)
        cascade_test_metrics = predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True)
        score_test = cascade_test_metrics[predictor.eval_metric.name]
        print(f'[INFO] cascade infer_time sec/row={pred_time_test_per_row}, {predictor.eval_metric.name}={score_test}')
        print(f'cascade pred_time_test={pred_time_test}')
        if infer_limit:
            assert pred_time_test_per_row <= (1 + cascade_time_diff) * infer_limit
        # clean up everything
        if delete_directory is True:
            shutil.rmtree(predictor.path, ignore_errors=True)  # Delete AutoGluon output directory to ensure runs' information has been removed.


# Helper functions for training models outside of predictors
class ModelFitHelper:
    @staticmethod
    def fit_and_validate_dataset(dataset_name, model, fit_args, sample_size=1000):
        directory_prefix = './datasets/'
        train_data, test_data, dataset_info = DatasetLoaderHelper.load_dataset(name=dataset_name, directory_prefix=directory_prefix)
        label = dataset_info['label']
        model, label_cleaner, feature_generator = ModelFitHelper.fit_dataset(train_data=train_data, model=model, label=label, fit_args=fit_args, sample_size=sample_size)
        if sample_size is not None and sample_size < len(test_data):
            test_data = test_data.sample(n=sample_size, random_state=0)

        X_test = test_data.drop(columns=[label])
        X_test = feature_generator.transform(X_test)

        model.predict(X_test)
        model.predict_proba(X_test)
        model.get_info()
        return model

    @staticmethod
    def fit_dataset(train_data, model, label, fit_args, sample_size=None):
        if sample_size is not None and sample_size < len(train_data):
            train_data = train_data.sample(n=sample_size, random_state=0)
        X = train_data.drop(columns=[label])
        y = train_data[label]

        problem_type = infer_problem_type(y)
        label_cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
        y = label_cleaner.transform(y)
        feature_generator = AutoMLPipelineFeatureGenerator()
        X = feature_generator.fit_transform(X, y)

        X, X_val, y, y_val = generate_train_test_split(X, y, problem_type=problem_type, test_size=0.2, random_state=0)

        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **fit_args)
        return model, label_cleaner, feature_generator


@pytest.fixture
def dataset_loader_helper():
    return DatasetLoaderHelper


@pytest.fixture
def fit_helper():
    return FitHelper


@pytest.fixture
def model_fit_helper():
    return ModelFitHelper
