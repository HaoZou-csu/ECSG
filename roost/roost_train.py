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


def input_parser():
    """
    parse input
    """
    parser = argparse.ArgumentParser(
        description=(
            "Roost - a Structure Agnostic Message Passing "
            "Neural Network for Inorganic Materials"
        )
    )

    # data inputs
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/datasets/expt-non-metals.csv",
        metavar="PATH",
        help="Path to main data set/training set",
    )
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-path",
        type=str,
        metavar="PATH",
        help="Path to independent validation set",
    )
    valid_group.add_argument(
        "--val-size",
        default=0.0,
        type=float,
        metavar="FLOAT",
        help="Proportion of data used for validation",
    )
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-path",
        type=str,
        metavar="PATH",
        help="Path to independent test set"
    )
    test_group.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        metavar="FLOAT",
        help="Proportion of data set for testing",
    )

    # data embeddings
    parser.add_argument(
        "--fea-path",
        type=str,
        default="./data/embeddings/matscholar-embedding.json",
        metavar="PATH",
        help="Element embedding feature path",
    )

    # dataloader inputs
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        metavar="INT",
        help="Number of data loading workers (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        "--bsize",
        default=128,
        type=int,
        metavar="INT",
        help="Mini-batch size (default: 128)",
    )
    parser.add_argument(
        "--data-seed",
        default=0,
        type=int,
        metavar="INT",
        help="Seed used when splitting data sets (default: 0)",
    )
    parser.add_argument(
        "--sample",
        default=1,
        type=int,
        metavar="INT",
        help="Sub-sample the training set for learning curves",
    )

    # optimiser inputs
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="INT",
        help="Number of training epochs to run (default: 100)",
    )
    parser.add_argument(
        "--loss",
        default="L1",
        type=str,
        metavar="STR",
        help="Loss function if regression (default: 'L1')",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Specifies whether to use hetroskedastic loss variants",
    )
    parser.add_argument(
        "--optim",
        default="AdamW",
        type=str,
        metavar="STR",
        help="Optimizer used for training (default: 'AdamW')",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        default=3e-4,
        type=float,
        metavar="FLOAT",
        help="Initial learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        default=1e-6,
        type=float,
        metavar="FLOAT [0,1]",
        help="Optimizer weight decay (default: 1e-6)",
    )

    # graph inputs
    parser.add_argument(
        "--elem-fea-len",
        default=64,
        type=int,
        metavar="INT",
        help="Number of hidden features for elements (default: 64)",
    )
    parser.add_argument(
        "--n-graph",
        default=3,
        type=int,
        metavar="INT",
        help="Number of message passing layers (default: 3)",
    )

    # ensemble inputs
    parser.add_argument(
        "--ensemble",
        default=1,
        type=int,
        metavar="INT",
        help="Number models to ensemble",
    )
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="STR",
        help="Name for sub-directory where models will be stored",
    )
    name_group.add_argument(
        "--data-id",
        default="roost",
        type=str,
        metavar="STR",
        help="Partial identifier for sub-directory where models will be stored",
    )
    parser.add_argument(
        "--run-id",
        default=0,
        type=int,
        metavar="INT",
        help="Index for model in an ensemble of models",
    )

    # restart inputs
    use_group = parser.add_mutually_exclusive_group()
    use_group.add_argument(
        "--fine-tune",
        type=str,
        metavar="PATH",
        help="Checkpoint path for fine tuning"
    )
    use_group.add_argument(
        "--transfer",
        type=str,
        metavar="PATH",
        help="Checkpoint path for transfer learning",
    )
    use_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint"
    )

    # task type
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--classification",
        action="store_true",
        help="Specifies a classification task"
    )
    task_group.add_argument(
        "--regression",
        action="store_true",
        help="Specifies a regression task"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the model/ensemble",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model/ensemble"
    )

    # misc
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA"
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log training metrics to tensorboard"
    )

    args = parser.parse_args(sys.argv[1:])

    if args.model_name is None:
        args.model_name = f"{args.data_id}_s-{args.data_seed}_t-{args.sample}"

    if args.regression:
        args.task = "regression"
    elif args.classification:
        args.task = "classification"
    else:
        args.task = "regression"

    args.device = (
        torch.device('cuda:0')
        if (not args.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return args

def train_model_5(df, cuda, n_fold, model_name = "roost", kfolds=10,  random_seed_3=123):
    args = input_parser()
    args.disable_cuda = cuda
    print(torch.cuda.is_available())
    args.device = 'cuda:0'
    print(f"The model will run on the {args.device} device")
    args.df = df
    args.train = True
    args.evaluate = True
    args.task = 'classification'
    args.test_path = pd.DataFrame([])
    args.batch_size = 2048
    args.epochs = 40

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
            args.model_name = model_name
            # train_cv_index.append(train_cv_X)
            # test_cv_index.append(test_cv_index)
            # main(**vars(args), kf_train=train_cv_X, kf_test=test_X,kf_val=None)
            main(**vars(args), kf_train=train_cv_X,  kf_val=val_cv_X)
        i = i+1


def train_single_model_5(path, cuda, train_X, test_X, val_X):
    args = input_parser()
    args.disable_cuda = cuda
    print(torch.cuda.is_available())
    args.device = 'cuda:0'
    print(f"The model will run on the {args.device} device")
    args.data_path = path
    args.train = True
    args.evaluate = True
    args.task = 'classification'

    model_name = "roost"


    main(**vars(args), kf_train=train_X, kf_test=test_X, kf_val=val_X)



def model_5_predict(pd, cuda, nfolds, model_name = "roost"):
    args = input_parser()
    args.disable_cuda = cuda
    print(torch.cuda.is_available())
    args.device = torch.device('cuda:0')
    print(f"The model will run on the {args.device} device")
    args.df = pd
    args.train = False
    args.evaluate = True
    args.task = 'classification'
    args.test_path = pd

    n = range(len(pd))

    for i in range(10):
        if i == nfolds:
            args.model_name = model_name
            main(**vars(args), predict=True)

def abla_train_model_5(path,cuda, kfolds=10, random_seed_1 = 123, random_seed_2 = 20201221):
    args = input_parser()
    args.disable_cuda = cuda
    print(torch.cuda.is_available())
    args.device = 'cuda:0'
    print(f"The model will run on the {args.device} device")
    args.data_path = path
    args.train = True
    args.evaluate = True
    args.task = 'classification'

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_seed_1)
    data = pd.read_csv(path)
    n = np.arange(len(data))
    train_X,test_X,_,_ = train_test_split(n,n,test_size=0.1,random_state=random_seed_2)
    train_X, _, train_y, _ = train_test_split(train_X, train_X, test_size=0.9, random_state=123456)#random_state not change, change test_size


    train_cv_index = []
    test_cv_index = []
    i = 0
    model_name = "abla_roost"

    for train, val in kf.split(train_X):
        if i == 0:
            train_cv_X = train_X[train]
            val_cv_X = train_X[val]
            args.model_name = model_name+"_"+str(i)
            # train_cv_index.append(train_cv_X)
            # test_cv_index.append(test_cv_index)
            # main(**vars(args), kf_train=train_cv_X, kf_test=test_X,kf_val=None)
            main(**vars(args), kf_train=train_cv_X, kf_test=test_X, kf_val=val_cv_X)
        i = i+1

def abla_train_model_5_stack(path,cuda, kfolds=10, random_seed_1 = 123, random_seed_2 = 20201221):
    args = input_parser()
    args.disable_cuda = cuda
    print(torch.cuda.is_available())
    args.device = 'cuda:0'
    print(f"The model will run on the {args.device} device")
    args.data_path = path
    args.train = True
    args.evaluate = True
    args.task = 'classification'

    kf = KFold(n_splits=kfolds, shuffle=True, random_state=random_seed_1)
    data = pd.read_csv(path)
    n = np.arange(len(data))
    train_X,test_X,_,_ = train_test_split(n,n,test_size=0.1,random_state=random_seed_2)
    train_X, _, train_y, _ = train_test_split(train_X, train_X, test_size=0.9, random_state=123456)#random_state not change, change test_size


    train_cv_index = []
    test_cv_index = []
    i = 0
    model_name = "abla_roost_stack"

    for train, val in kf.split(train_X):

        train_cv_X = train_X[train]
        val_cv_X = train_X[val]
        args.model_name = model_name+"_"+str(i)
        args.ensemble = 1
        # train_cv_index.append(train_cv_X)
        # test_cv_index.append(test_cv_index)
        # main(**vars(args), kf_train=train_cv_X, kf_test=test_X,kf_val=None)
        main(**vars(args), kf_train=train_cv_X, kf_test=test_X, kf_val=val_cv_X)
        i = i+1


if __name__ == "__main__":


    args = input_parser()
    print(torch.cuda.is_available())
    print(f"The model will run on the {args.device} device")
    # main(**vars(args))
    n = np.arange(85014)
    n,test_X,_,_ = train_test_split(n,n,test_size=0.1,random_state=20201221)
    train_X, _, _, _ = train_test_split(n, n, test_size=0.9, random_state=123456)
    kf = KFold(n_splits=10, shuffle=True, random_state=123)
    train_cv_index = []
    test_cv_index = []
    i = 0
    model_name = "roost"

    for train, val in kf.split(train_X):
        train_cv_X = train_X[train]
        val_cv_X = train_X[val]
        args.model_name = model_name+"_"+str(i)
        # train_cv_index.append(train_cv_X)
        # test_cv_index.append(test_cv_index)
        # main(**vars(args), kf_train=train_cv_X, kf_test=test_X,kf_val=None)
        main(**vars(args), kf_train=train_cv_X, kf_test=test_X, kf_val=val_cv_X)

        i = i+1

#--data-path ./data/datasets/mp_data_Roost.csv --train --evaluate --classification