
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
    for infer_limit_batch_size, sim_time_diff in [(10000, 0.2), (100000, 0.1)]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds, simulation_time_diff=sim_time_diff,
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
    cascade = ['KNeighbors_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'NeuralNetTorch_BAG_L2_FULL', 'WeightedEnsemble_L3_FULL']
    cascade_thresholds = [0.90, 0.75, 0.75]
    for infer_limit_batch_size, sim_time_diff in [(10000, 0.3), (100000, 0.1)]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds, simulation_time_diff=sim_time_diff,
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
    for infer_limit_batch_size, sim_time_diff in [(10000, 0.3), (100000, 0.1)]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds, simulation_time_diff=sim_time_diff,
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
    cascade = ['KNeighbors_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'NeuralNetTorch_BAG_L2_FULL', 'WeightedEnsemble_L3_FULL']
    cascade_thresholds = [0.90, 0.75, 0.75]
    for infer_limit_batch_size, sim_time_diff in [(10000, 0.3), (100000, 0.1)]:
        fit_helper.check_cascade_speed_accuracy_simulation(dataset_name, fit_args, 
                cascade, cascade_thresholds, simulation_time_diff=sim_time_diff,
                infer_limit_batch_size=infer_limit_batch_size,
                train_sample_size=3000)


def test_fit_cascade_meet_infer_limit_binary_F2SP(fit_helper):
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'adult'
    # Test Fast-to-Slow default params 
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
    }
    infer_limit_batch_size = 10000
    infer_limit = None
    fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                             hyperparameter_cascade=hyperparameter_cascade)
    # Test Fast-to-Slow with Random Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'Random',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': {},
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-5), (100000, 5e-6)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade)
    # Test Fast-to-Slow with TPE Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'TPE',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': {},
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-5), (100000, 5e-6)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade)


def test_fit_cascade_meet_infer_limit_binary_F2SP_hq(fit_helper):
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
    dataset_name = 'adult'
    # Test Fast-to-Slow default params 
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
    }
    infer_limit_batch_size = 10000
    infer_limit = None
    fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                             hyperparameter_cascade=hyperparameter_cascade)
    # Test Fast-to-Slow with Random Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'Random',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-5), (100000, 5e-6)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                hyperparameter_cascade=hyperparameter_cascade)
    # Test Fast-to-Slow with TPE Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'TPE',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-5), (100000, 5e-6)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade)


def test_fit_cascade_meet_infer_limit_multiclass_F2SP(fit_helper):
    fit_args = dict(
        hyperparameters={
            KNNModel: {},
            LGBModel: {},
            TabularNeuralNetTorchModel: {},
        },
    )
    dataset_name = 'covertype_small'
    # Test Fast-to-Slow default params 
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
    }
    infer_limit_batch_size = 10000
    infer_limit = None
    fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                             hyperparameter_cascade=hyperparameter_cascade,
                                             train_sample_size=3000)
    # Test Fast-to-Slow with Random Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'Random',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-4), (100000, 1e-5)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade,
                                                 train_sample_size=3000)
    # Test Fast-to-Slow with TPE Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'TPE',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-4), (100000, 1e-5)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade,
                                                 train_sample_size=3000)


def test_fit_cascade_meet_infer_limit_multiclass_F2SP_hq(fit_helper):
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
    # Test Fast-to-Slow default params 
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
    }
    infer_limit_batch_size = 10000
    infer_limit = None
    fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                             hyperparameter_cascade=hyperparameter_cascade,
                                             train_sample_size=3000)
    # Test Fast-to-Slow with Random Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'Random',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-4), (100000, 1e-5)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                 hyperparameter_cascade=hyperparameter_cascade,
                                                 train_sample_size=3000)
    # Test Fast-to-Slow with TPE Searcher
    hyperparameter_cascade = {
            'cascade_algo': 'F2S+',
            'num_trials': 200,
            'searcher': 'TPE',
            'hpo_score_func': 'eval_metric',
            'hpo_score_func_kwargs': None,
    }
    for infer_limit_batch_size, infer_limit in [(10000, 1e-4), (100000, 1e-5)]:
        fit_helper.check_fit_cascade_infer_limit(dataset_name, fit_args, infer_limit, infer_limit_batch_size, 
                                                hyperparameter_cascade=hyperparameter_cascade,
                                                train_sample_size=3000)


def test_cascade_config_pruning(fit_helper):
    from autogluon.tabular.predictor.cascade_do_no_harm import CascadeConfig, prune_cascade_config
    from autogluon.core.constants import BINARY, MULTICLASS
    # case 1: no need to prune
    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(0.9, 0.75), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert config.model == ret_config.model
    assert config.thresholds == ret_config.thresholds

    # case 2: prune member models
    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(1.0, 0.75), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL')
    assert ret_config.thresholds == (0.75, )

    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(0.85, 1.0), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('KNeighborsUnif_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL')
    assert ret_config.thresholds == (0.85, )

    config = CascadeConfig(
        model=('LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(1.0, ), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=MULTICLASS)
    assert ret_config.model == ('WeightedEnsemble_L2_FULL', )
    assert ret_config.thresholds == ()

    ## two members to prune
    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(1.0, 1.0), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('WeightedEnsemble_L2_FULL', )
    assert ret_config.thresholds == ()

    # case 3: early exit
    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(0.5, 0.95), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('KNeighborsUnif_BAG_L1_FULL',)
    assert ret_config.thresholds == ()

    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(0.0, 0.5), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=MULTICLASS)
    assert ret_config.model == ('KNeighborsUnif_BAG_L1_FULL',)
    assert ret_config.thresholds == ()

    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(0.85, 0.5), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL')
    assert ret_config.thresholds == (0.85, )

    config = CascadeConfig(
        model=('KNeighborsUnif_BAG_L1_FULL', 'LightGBM_BAG_L1_FULL', 'RandomForest_BAG_L1_FULL', 'WeightedEnsemble_L2_FULL'), 
        thresholds=(1.0, 0.5, 1.0), pred_time_val=0.0144, score_val=0.7808, hpo_score=0.7808, 
        hpo_func_name='ACCURACY', 
        )
    ret_config = prune_cascade_config(config, problem_type=BINARY)
    assert ret_config.model == ('LightGBM_BAG_L1_FULL', )
    assert ret_config.thresholds == ()