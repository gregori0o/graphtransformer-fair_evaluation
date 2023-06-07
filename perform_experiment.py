import itertools
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

from data.load_data import DatasetName, GraphsDataset, load_indexes
from evaluation_config import K_FOLD, R_EVALUATION
from train_graph_transformer import train_graph_transformer
from utils import NpEncoder

configurations = {
    DatasetName.ZINC: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.DD: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.NCI1: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.PROTEINS: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.ENZYMES: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.IMDB_BINARY: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.IMDB_MULTI: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.REDDIT_BINARY: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.REDDIT_MULTI: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
    DatasetName.COLLAB: {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    },
}


def get_all_params(param_grid):
    return [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]


def perform_experiment(dataset_name):
    # config = configurations.get(dataset_name)
    # if config is None:
    #     raise ValueError(f"There is no configuration for {dataset_name} dataset.")

    config = {
        "config_path": "configs/zinc_config.json",
        "net_parans_grid": {},
        "params_grid": {},
        "tune_hyperparameters": False,
    }

    with open(config["config_path"], "r") as f:
        train_config = json.load(f)
    train_config["out_dir"] = f"out/{dataset_name.value}/"

    # load indexes
    indexes = load_indexes(dataset_name)
    assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    # loop over splits
    scores = []
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")

        ## get best model for train data
        if config.get("tune_hyperparameters"):
            best_acc = 0
            best_params = ({}, {})
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.2)
            params = get_all_params(config["params_config"])
            net_params = get_all_params(config["net_params"])
            for param in params:
                train_config["params"].update(param)
                for net_param in net_params:
                    train_config["net_params"].update(net_param)
                    dataset = GraphsDataset(
                        dataset_name, train_idx, val_idx, val_idx
                    )  # test_idx <- val_idx
                    acc = train_graph_transformer(dataset, train_config)
                    if acc > best_acc:
                        best_acc = acc
                        best_params = (param.copy(), net_param.copy())

            train_config["params"].update(best_params[0])
            train_config["net_params"].update(best_params[1])

        # evaluate model R times
        scores_r = 0
        test_idx = fold["test"]
        for _ in range(R_EVALUATION):
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.2)
            dataset = GraphsDataset(dataset_name, train_idx, val_idx, test_idx)
            acc = train_graph_transformer(dataset, train_config)
            scores_r += acc
        scores_r /= R_EVALUATION
        print(f"MEAN SCORE = {scores_r} in FOLD {i}")
        scores.append(scores_r)

    # evaluate model
    mean = np.mean(scores)
    std = np.std(scores)

    # score is MAE for regression and ACC for other
    print(f"Evaluation of model on {dataset_name}")
    print(f"Mean score: {mean}")
    print(f"STD score: {std}")

    train_config["dataset_name"] = dataset_name.value
    train_config["mean_score"] = mean
    train_config["std_score"] = std
    del train_config["net_params"]["device"]
    dumped = json.dumps(train_config, cls=NpEncoder)
    with open(
        f"results/result_GT_{dataset_name.value}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}.json",
        "w",
    ) as f:
        f.write(dumped)


if __name__ == "__main__":
    # for dataset_name in DatasetName:
    perform_experiment(DatasetName.ENZYMES)
