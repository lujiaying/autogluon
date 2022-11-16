import time

from autogluon.tabular import TabularPredictor, TabularDataset


if __name__ == '__main__':
    path_prefix = 'https://autogluon.s3.amazonaws.com/datasets/CoverTypeMulticlassClassification/'
    path_train = path_prefix + 'train_data.csv'
    path_test = path_prefix + 'test_data.csv'

    label = 'Cover_Type'
    sample = 50000  # Number of rows to use to train
    train_data = TabularDataset(path_train)

    if sample is not None and (sample < len(train_data)):
        train_data = train_data.sample(n=sample, random_state=0).reset_index(drop=True)

    test_data = TabularDataset(path_test)

    fit_kwargs = dict(
        train_data=train_data,
        hyperparameters={
            'LR': {},
            'GBM': {},
            'KNN': {},
        },
    )

    predictor = TabularPredictor(
        label=label,
        eval_metric='log_loss',
    )
    predictor.fit(**fit_kwargs)
    predictor.persist_models('all')

    leaderboard = predictor.leaderboard(test_data)

    # get normal AG result
    ts = time.time()
    pred_proba = predictor.predict_proba(test_data, model='WeightedEnsemble_L2')
    te = time.time()
    print('--------')
    print('Normal: ')
    print(predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True))
    print(f'infer time: {te - ts}s')
    print('--------')

    # get cascade AG result
    # == use default setting, which means infer_limit=None, algorithm=F2S+/ag_goodness
    chosen_cascade_conf = predictor.fit_cascade(train_data)
    ts = time.time()
    _, pred_proba = predictor.do_infer_with_cascade_conf(chosen_cascade_conf, test_data)
    te = time.time()
    print('--------')
    print('Cascade (default): ')
    print(predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True))
    print(f'infer time: {te - ts}s')
    print('--------')

    # == specify infer_limit and declaring that we want max log_loss when satisfying infer_limit
    chosen_cascade_conf = predictor.fit_cascade(
        train_data,
        infer_limit=2,
        infer_limit_batch_size=10000,
        hyperparameter_cascade=dict(cascade_algo='F2S+', hpo_score_func="eval_metric"),
        )
    ts = time.time()
    _, pred_proba = predictor.do_infer_with_cascade_conf(chosen_cascade_conf, test_data)
    te = time.time()
    print('--------')
    print('Cascade (infer_limit=2 sec): ')
    print(predictor.evaluate_predictions(y_true=test_data[label], y_pred=pred_proba, silent=True))
    print(f'infer time: {te - ts}s')
    print('--------')


    """ Below are example outputs
    --------
    Normal: 
    {'log_loss': -0.25209540888145077, 'accuracy': 0.9059490718828257, 'balanced_accuracy': 0.8154290346790535, 'mcc': 0.8480809391135653}
    infer time: 13.982260704040527s
    --------
    
    --------
    Cascade (default): 
    {'log_loss': -0.6297591899218917, 'accuracy': 0.7241981704431039, 'balanced_accuracy': 0.5121288591825748, 'mcc': 0.5482599937950497}
    infer time: 0.6160697937011719s
    --------

    --------
    Cascade (infer_limit=2 sec): 
    {'log_loss': -0.26133042887283264, 'accuracy': 0.9061470013682952, 'balanced_accuracy': 0.8147791755269156, 'mcc': 0.8484080238307692}
    infer time: 7.726653337478638s
    --------

    """