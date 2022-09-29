import os
import json
import random
import argparse
from typing import Optional, Callable, Tuple
from functools import partial
import time
from datetime import datetime

import git
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor, FeatureMetadata, __version__
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular.predictor.cascade_do_no_harm import get_all_predecessor_model_names
from autogluon.core.utils.time import sample_df_for_time_func


def get_parquet_path(train_dir: str) -> str:
    train_fname = os.listdir(train_dir)
    assert len(train_fname) == 1  # list for temp
    train_fname = train_fname[0]
    train_fpath = os.path.join(train_dir, train_fname)
    return train_fpath

def image_id_to_path(image_dir: str, image_path_suffix: str, image_id: str
        ) -> Optional[str]:
    if isinstance(image_id, str):
        image_path = os.path.join(image_dir, f'{image_id}.{image_path_suffix}')
        if os.path.exists(image_path):
            return image_path
        else:
            return None
    else:
        return None


def load_cpp_dataset(fpath: str, image_map_func: Callable[[str], str]) -> Tuple[TabularDataset, FeatureMetadata]:
    dataset = TabularDataset(fpath)
    image_col = 'image_id'
    # add image column
    dataset[image_col] = dataset[image_col].apply(image_map_func)
    feature_metadata = FeatureMetadata.from_df(dataset)
    feature_metadata = feature_metadata.add_special_types({image_col: ['image_path']})
    return dataset, feature_metadata


def main(args: argparse.Namespace):
    random.seed(args.seed)
    image_id_to_path_cpp = partial(image_id_to_path, args.cpp_img_dir, 'jpg')
    # traversal
    for cpp_session in os.listdir(args.cpp_dir):
        if args.session_names != None and cpp_session not in args.session_names:
            print(f'DEBUG skip {cpp_session}')
        session_ts = time.time()
        train_fpath = get_parquet_path(os.path.join(args.cpp_dir, cpp_session, 'train'))
        test_fpath = get_parquet_path(os.path.join(args.cpp_dir, cpp_session, 'test'))
        # load data
        train_data, feature_metadata = load_cpp_dataset(train_fpath, image_id_to_path_cpp)
        test_data, _ = load_cpp_dataset(test_fpath, image_id_to_path_cpp)
        # prepare fit kwargs
        fit_kwargs = dict(
            train_data=train_data,
            hyperparameters=get_hyperparameter_config(args.fit_hyperparameter_config),
            presets=args.fit_preset,
            time_limit=args.fit_time_limit,
            infer_limit=args.fit_infer_limit,
            infer_limit_batch_size=args.infer_limit_batch_size,
            feature_metadata=feature_metadata,
        )
        exp_result_save_session_dir = os.path.join(args.exp_result_save_dir, cpp_session)
        if not os.path.exists(exp_result_save_session_dir):
            os.makedirs(exp_result_save_session_dir)
        predictor = TabularPredictor(
            label='label',
            eval_metric=args.eval_metric,
            path=os.path.join(exp_result_save_session_dir, 'models'),
        )
        ts = time.time()
        predictor.fit(**fit_kwargs)
        te = time.time()
        training_duration = te - ts
        predictor.persist_models()
        # get test duration
        ts = time.time()
        score = predictor.evaluate(test_data)
        te = time.time()
        predict_duration = te - ts
        # get genuine test duration
        test_data_sampled = sample_df_for_time_func(df=test_data, sample_size=args.infer_limit_batch_size, 
                                max_sample_size=args.infer_limit_batch_size)
        ts = time.time()
        _ = predictor.evaluate(test_data_sampled)
        te = time.time()
        predict_genuine_duration = te - ts
        duration = te - session_ts
        repo = git.Repo(search_parent_directories=True)
        models_ensemble_count = len(get_all_predecessor_model_names(predictor, predictor.get_model_best()))
        result = dict(
            id=f'CPP{cpp_session}',
            task=cpp_session,
            framework='AGv053_Aug31_high',  # TODO: change when using other
            constraint=f'{int(args.fit_time_limit/3600)}h8c_nd4g.2xlarge',
            fold=-1,
            type='binary',
            metric=args.eval_metric,
            mode='local',
            version=__version__,
            params=json.dumps(args.__dict__),
            app_version=json.dumps([repo.active_branch.name, repo.git.rev_parse(repo.head.object.hexsha, short=7)]),
            utc=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S'),
            duration=round(duration, 1),
            training_duration=round(training_duration, 1),
            predict_duration=round(predict_duration, 1),
            models_count=len(predictor.get_model_names()),
            seed=args.seed,
            info=None,
            acc=score['accuracy'],
            auc=score['roc_auc'],
            balacc=score['balanced_accuracy'],
            logloss=-score['log_loss'],
            f1=score['f1'],
            models_ensemble_count=models_ensemble_count,
            predict_genuine_duration=round(predict_genuine_duration),
        )
        result_df = pd.DataFrame.from_records([result])
        result_dir = os.path.join(exp_result_save_session_dir, 'scores')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_df.to_csv(os.path.join(result_dir, 'result.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Exp arguments to set")
    # required args
    parser.add_argument('--exp_result_save_dir', type=str, required=True)
    # optional args
    parser.add_argument('--cpp_dir', type=str, default='datasets/cpp_research_corpora/2021_60datasets')
    parser.add_argument('--cpp_img_dir', type=str, default='/home/ec2-user/autogluon/datasets/cpp_research_corpora/2021_60datasets_imgs_raw')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fit_hyperparameter_config', type=str, default='multimodal')
    parser.add_argument('--fit_preset', type=str, default='high_quality')
    parser.add_argument('--fit_time_limit', type=int, default=14400)
    parser.add_argument('--fit_infer_limit', type=float, default=None)
    parser.add_argument('--infer_limit_batch_size', type=int, default=10000)
    parser.add_argument('--eval_metric', type=str, default='acc')
    parser.add_argument('--session_names', type=str, nargs='+', default=None)

    args = parser.parse_args()
    print(f'Exp arguments: {args}')
    main(args)
