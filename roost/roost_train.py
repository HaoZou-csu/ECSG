import os
import sys
import argparse

import pandas as pd
import torch
from sklearn.model_selection import train_test_split as split
import numpy as np
from sklearn.model_selection import train_test_split, KFold

from os.path import dirname, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)

from roost.model import Roost
from roost.data import CompositionData, collate_batch, CompositionData_SG
from roost.utils import (train_ensemble,
    results_regression,
    results_classification,
    train_results_classification,
    val_results_classification
                               )

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(
    df,# mp_data_Roost pandas Dataframe
    fea_path,
    task,
    loss,
    robust,
    ensemble,
    model_name="roost",
    elem_fea_len=64,
    n_graph=3,
    run_id=1,
    data_seed=42,
    epochs=100,
    patience=None,
    log=True,
    sample=1,
    test_size=0.2,
    test_path=None,
    val_size=0.0,
    val_path=None,
    resume=None,
    fine_tune=None,
    transfer=None,
    train=True,
    evaluate=True,
    predict = False,
    optim="AdamW",
    learning_rate=3e-4,
    momentum=0.9,
    weight_decay=1e-6,
    batch_size=128,
    workers=0,
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu"),
    kf_train=None,
    kf_val=None,
    kf_test=None,
    **kwargs,
):
    assert evaluate or train, (
        "No task given - Set at least one of 'train' or 'evaluate' kwargs as True"
    )
    assert task in ["regression", "classification"], (
        "Only 'regression' or 'classification' allowed for 'task'"
    )

    if not test_path.empty:
        test_size = 0.0

    if not (not test_path.empty and val_path):
        assert test_size + val_size < 1.0, (
            f"'test_size'({test_size}) "
            f"plus 'val_size'({val_size}) must be less than 1"
        )

    if ensemble > 1 and (fine_tune or transfer):
        raise NotImplementedError(
            "If training an ensemble with fine tuning or transfering"
            " options the models must be trained one by one using the"
            " run-id flag."
        )

    assert not (fine_tune and transfer), (
        "Cannot fine-tune and" " transfer checkpoint(s) at the same time."
    )

    dataset = CompositionData_SG(df=df, fea_path=fea_path, task=task)
    n_targets = dataset.n_targets
    elem_emb_len = dataset.elem_emb_len

    train_idx = list(range(len(dataset)))

    if evaluate:
        if kf_test is not None:
            test_set = torch.utils.data.Subset(dataset, kf_test)
        else:
            if not test_path.empty:
                print("using independent test set")
                test_set = CompositionData_SG(
                    df=test_path, fea_path=fea_path, task=task
                )
                test_set = torch.utils.data.Subset(test_set, range(len(test_set)))
            elif test_size == 0.0:
                raise ValueError("test-size must be non-zero to evaluate model")
            else:
                print(f"using {test_size} of training set as test set")
                train_idx, test_idx = split(
                    train_idx, random_state=data_seed, test_size=test_size
                )
                test_set = torch.utils.data.Subset(dataset, test_idx)

    if train:
        if val_path:
            print(f"using independent validation set: {val_path}")
            val_set = CompositionData_SG(df=val_path, fea_path=fea_path, task=task)
            val_set = torch.utils.data.Subset(val_set, range(len(val_set)))
        else:
            if val_size == 0.0 and evaluate:
                print("No validation set used, using test set for evaluation purposes")
                # NOTE that when using this option care must be taken not to
                # peak at the test-set. The only valid model to use is the one
                # obtained after the final epoch where the epoch count is
                # decided in advance of the experiment.
                val_set = test_set
            elif val_size == 0.0:
                val_set = None
            else:
                print(f"using {val_size} of training set as validation set")
                train_idx, val_idx = split(
                    train_idx, random_state=data_seed, test_size=val_size / (1 - test_size),
                )
                val_set = torch.utils.data.Subset(dataset, val_idx)

        # train_set = torch.utils.data.Subset(dataset, train_idx[0::sample])
        if kf_train is not None:
            train_set = torch.utils.data.Subset(dataset, kf_train)
        if kf_val is not None:
            val_set = torch.utils.data.Subset(dataset, kf_val)

    data_params = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": collate_batch,
    }

    setup_params = {
        "loss": loss,
        "optim": optim,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
    }

    restart_params = {
        "resume": resume,
        "fine_tune": fine_tune,
        "transfer": transfer,
    }

    model_params = {
        "task": task,
        "robust": robust,
        "n_targets": n_targets,
        "elem_emb_len": elem_emb_len,
        "elem_fea_len": elem_fea_len,
        "n_graph": n_graph,
        "elem_heads": 3,
        "elem_gate": [256],
        "elem_msg": [256],
        "cry_heads": 3,
        "cry_gate": [256],
        "cry_msg": [256],
        "out_hidden": [1024, 512, 256, 128, 64],
    }

    os.makedirs(f"models/{model_name}/", exist_ok=True)

    if log:
        os.makedirs("runs/", exist_ok=True)

    os.makedirs("results/", exist_ok=True)

    if train:
        train_ensemble(
            model_class=Roost,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            epochs=epochs,
            patience=patience,
            train_set=train_set,
            val_set=val_set,
            log=log,
            data_params=data_params,
            setup_params=setup_params,
            restart_params=restart_params,
            model_params=model_params,
        )

        if evaluate:

            data_reset = {
                "batch_size": 16 * batch_size,  # faster model inference
                "shuffle": False,  # need fixed data order due to ensembling
            }
            data_params.update(data_reset)

            if task == "regression":
                results_regression(
                    model_class=Roost,
                    model_name=model_name,
                    run_id=run_id,
                    ensemble_folds=ensemble,
                    test_set=test_set,
                    data_params=data_params,
                    robust=robust,
                    device=device,
                    eval_type="checkpoint",
                )
            elif task == "classification":
                train_results_classification(
                    model_class=Roost,
                    model_name=model_name,
                    run_id=run_id,
                    ensemble_folds=ensemble,
                    test_set=train_set,
                    data_params=data_params,
                    robust=robust,
                    device=device,
                    eval_type="checkpoint",
                )
                val_results_classification(
                    model_class=Roost,
                    model_name=model_name,
                    run_id=run_id,
                    ensemble_folds=ensemble,
                    test_set=val_set,
                    data_params=data_params,
                    robust=robust,
                    device=device,
                    predict=predict,
                    eval_type="checkpoint",
                )
                results_classification(
                    model_class=Roost,
                    model_name=model_name,
                    run_id=run_id,
                    ensemble_folds=ensemble,
                    test_set=test_set,
                    data_params=data_params,
                    robust=robust,
                    device=device,
                    eval_type="checkpoint",
                )
    else:
        val_results_classification(
            model_class=Roost,
            model_name=model_name,
            run_id=run_id,
            ensemble_folds=ensemble,
            test_set=test_set,
            data_params=data_params,
            robust=robust,
            device=device,
            predict=predict,
            eval_type="checkpoint",
        )



model_5_vars = {'data_path': '../data/datasets/expt-non-metals.csv', 'val_path': None, 'val_size': 0.0, 'test_path': None,
     'test_size': 0.2, 'fea_path': './data/embeddings/matscholar-embedding.json', 'workers': 0, 'batch_size': 128,
     'data_seed': 0, 'sample': 1, 'epochs': 100, 'loss': 'L1', 'robust': False, 'optim': 'AdamW',
     'learning_rate': 0.0003, 'momentum': 0.9, 'weight_decay': 1e-06, 'elem_fea_len': 64, 'n_graph': 3, 'ensemble': 1,
     'model_name': 'roost_s-0_t-1', 'data_id': 'roost', 'run_id': 0, 'fine_tune': None, 'transfer': None,
     'resume': False, 'classification': False, 'regression': False, 'evaluate': False, 'train': False,
     'disable_cuda': False, 'log': False, 'task': 'regression', 'device': 'cuda:0'}




def train_model_5(df, cuda, device, epoch, batch_size, n_fold, model_name = "roost", kfolds=10,  random_seed_3=123):
    model_5_vars['disable_cuda'] = cuda
    print(torch.cuda.is_available())
    model_5_vars['device'] = device
    print(f"The model will run on the {model_5_vars['device']} device")
    model_5_vars['df'] = df
    model_5_vars['train'] = True
    model_5_vars['evaluate'] = True
    model_5_vars['task'] = 'classification'
    model_5_vars['test_path'] = pd.DataFrame([])
    model_5_vars['batch_size'] = batch_size
    model_5_vars['epochs'] = epoch

    n = np.arange(len(df))
    train_X = n

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_seed_3)
    train_cv_index = []
    test_cv_index = []
    i = 0

    for train, val in kf.split(train_X):
        if i == n_fold:
            train_cv_X = train_X[train]
            val_cv_X = train_X[val]
            model_5_vars['model_name'] = model_name
            # train_cv_index.append(train_cv_X)
            # test_cv_index.append(test_cv_index)
            # main(**vars(args), kf_train=train_cv_X, kf_test=test_X,kf_val=None)
            main(**model_5_vars, kf_train=train_cv_X,  kf_val=val_cv_X)
        i = i+1



def model_5_predict(pd, cuda, nfolds, model_name = "roost"):
    model_5_vars['disable_cuda'] = cuda
    print(torch.cuda.is_available())
    model_5_vars['device'] = 'cuda:0'
    print(f"The model will run on the {model_5_vars['device']} device")
    model_5_vars['df'] = pd
    model_5_vars['train'] = False
    model_5_vars['evaluate'] = True
    model_5_vars['task'] = 'classification'
    model_5_vars['test_path'] = pd
    model_5_vars['batch_size'] = 2048
    model_5_vars['epochs'] = 100
    pd['target'] = np.random.randint(2, size=len(pd))

    n = range(len(pd))

    for i in range(10):
        if i == nfolds:
            model_5_vars['model_name'] = model_name
            main(**model_5_vars, predict=True)


if __name__ == "__main__":



    print(torch.cuda.is_available())
