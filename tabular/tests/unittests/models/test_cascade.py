
from autogluon.tabular.models import LGBModel, TabularNeuralNetTorchModel, KNNModel


def test_cascade_binary(fit_helper):
    fit_args = dict(
        hyperparameters={
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'adult'
    cascade = ['LightGBM', 'WeightedEnsemble_L2']
    fit_helper.fit_and_validate_dataset_with_cascade(dataset_name=dataset_name, fit_args=fit_args, cascade=cascade, model_count=2)


def test_cascade_multiclass(fit_helper):
    fit_args = dict(
        hyperparameters={
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'covertype_small'
    cascade = ['LightGBM', 'WeightedEnsemble_L2']
    fit_helper.fit_and_validate_dataset_with_cascade(dataset_name=dataset_name, fit_args=fit_args, cascade=cascade, model_count=2)


def test_cascade_infer_simulation_binary(fit_helper):
    """
    Test whether implemented simulation align with real execution
    """
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'adult'
    cascade = ['KNeighbors', 'LightGBM', 'WeightedEnsemble_L2']
    cascade_thresholds = [0.90, 0.75]
    for infer_limit_batch_size in [10000, 100000]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds,
                infer_limit_batch_size=infer_limit_batch_size)


def test_cascade_infer_simulation_binary_hq(fit_helper):
    """
    Test whether implemented simulation align with real execution
    Models are trained on high-quality mode
    """
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
        presets='high_quality',
        num_bag_folds=3,
        num_bag_sets=2,
        num_stack_levels=1, 
    )
    dataset_name = 'adult'
    #cascade = ['KNeighbors_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL']
    cascade = ['KNeighbors_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'NeuralNetTorch_BAG_L2_FULL', 'WeightedEnsemble_L3_FULL']
    cascade_thresholds = [0.90, 0.75, 0.75]
    #for infer_limit_batch_size in [10000, 100000]:
    for infer_limit_batch_size in [100000]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds,
                infer_limit_batch_size=infer_limit_batch_size)


def test_cascade_infer_simulation_multiclass(fit_helper):
    """
    Test whether implemented simulation align with real execution
    """
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'covertype_small'
    cascade = ['KNeighbors', 'LightGBM', 'WeightedEnsemble_L2']
    cascade_thresholds = [0.8, 0.7]
    for infer_limit_batch_size in [10000, 100000]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds,
                infer_limit_batch_size=infer_limit_batch_size,
                train_sample_size=3000)


def test_cascade_infer_simulation_multiclass_hq(fit_helper):
    """
    Test whether implemented simulation align with real execution
    Models are trained on high-quality mode
    """
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
        presets='high_quality',
        num_bag_folds=2,
        num_bag_sets=2,
    )
    dataset_name = 'covertype_small'
    cascade = ['KNeighbors_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL']
    cascade_thresholds = [0.8, 0.7]
    for infer_limit_batch_size in [10000, 100000]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds,
                infer_limit_batch_size=infer_limit_batch_size,
                train_sample_size=3000)