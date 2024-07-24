import itertools
import json
import os
import time

import numpy as np
import optuna
import torch
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data.load_data import DatasetName, GraphsDataset, load_indexes
from evaluation_config import K_FOLD, R_EVALUATION
from time_measure import time_measure
from train_graph_transformer import train_graph_transformer
from utils import NpEncoder

experiment_name = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")


special_params = {
    DatasetName.DD: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.NCI1: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.ENZYMES: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.PROTEINS: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.IMDB_BINARY: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.IMDB_MULTI: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.REDDIT_BINARY: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.REDDIT_MULTI: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.COLLAB: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.MOLHIV: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.WEB: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
    DatasetName.MUTAGEN: {
        "params": {
            "epochs": 100,
        },
        "net_params": {},
    },
}


def prepare_dataset(dataset, train_config):
    if train_config["net_params"]["lap_pos_enc"]:
        # st = time.time()
        # print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(
            train_config["net_params"]["pos_enc_dim"]
        )
        # print("Time LapPE:", time.time() - st)

    if train_config["net_params"]["wl_pos_enc"]:
        # st = time.time()
        # print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        # print("Time WL PE:", time.time() - st)

    if train_config["net_params"]["full_graph"]:
        # st = time.time()
        # print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        # print("Time taken to convert to full graphs:", time.time() - st)

    if train_config["net_params"]["self_loop"]:
        # st = time.time()
        # print("[!] Converting the given graphs, adding self loops..")
        dataset._add_self_loops()
        # print("Time taken to add self loops:", time.time() - st)


def find_best_params(train_config, loaded_dataset, dataset_name, fold):
    def optuna_objective(trial):
        ### Definition of the search space ###
        train_config["params"]["init_lr"] = trial.suggest_float(
            "init_lr", 1e-6, 1e-3, log=True
        )
        train_config["params"]["lr_reduce_factor"] = trial.suggest_float(
            "lr_reduce_factor", 0.0, 1.0
        )
        train_config["params"]["weight_decay"] = trial.suggest_float(
            "weight_decay", 0.0, 1.0
        )
        train_config["net_params"]["in_feat_dropout"] = trial.suggest_float(
            "in_feat_dropout", 0.0, 1.0
        )
        train_config["net_params"]["dropout"] = trial.suggest_float("dropout", 0.0, 1.0)
        train_config["net_params"]["L"] = trial.suggest_int("L", 5, 20)
        ### End ###
        train_config["trial"] = trial

        # if there is "lap_pos_enc" or "wl_pos_enc" or "full_graph" or "self_loop" in search space
        # then needs_new_dataset = True
        needs_new_dataset = False

        if needs_new_dataset:
            dataset = GraphsDataset(dataset_name)
            prepare_dataset(dataset, train_config)
            dataset.upload_indexes(train_idx, val_idx, val_idx)  # test_idx <- val_idx
        else:
            dataset = loaded_dataset

        acc = train_graph_transformer(dataset, train_config)[0]["accuracy"]
        return acc

    train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
    loaded_dataset.upload_indexes(train_idx, val_idx, val_idx)

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=10, timeout=None)
    train_config["trial"] = None

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    best_params = {"params": {}, "net_params": {}}
    for key, value in trial.params.items():
        if key in train_config["params"]:
            best_params["params"][key] = value
        else:
            best_params["net_params"][key] = value

    return best_params, trial.value


def get_prediction(model, device, data_loader):
    model.eval()
    list_predictions = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata["feat"].to(device)
            batch_e = batch_graphs.edata["feat"].to(device)
            batch_targets = batch_targets.to(device)
            try:
                batch_lap_pos_enc = batch_graphs.ndata["lap_pos_enc"].to(device)
            except:
                batch_lap_pos_enc = None

            try:
                batch_wl_pos_enc = batch_graphs.ndata["wl_pos_enc"].to(device)
            except:
                batch_wl_pos_enc = None

            batch_scores = model.forward(
                batch_graphs, batch_x, batch_e, batch_lap_pos_enc, batch_wl_pos_enc
            )
            list_predictions.append(batch_scores.detach().argmax(dim=1).cpu().numpy())
    predictions = np.concatenate(list_predictions)
    return predictions


def perform_experiment(dataset_name):
    # config = configurations.get(dataset_name)
    # if config is None:
    #     raise ValueError(f"There is no configuration for {dataset_name} dataset.")

    config = {
        "config_path": "configs/universal_config.json",
        "tune_hyperparameters": False,
    }

    with open(config["config_path"], "r") as f:
        train_config = json.load(f)
    train_config["params"].update(special_params[dataset_name]["params"])
    train_config["net_params"].update(special_params[dataset_name]["net_params"])
    train_config["out_dir"] = f"out/{dataset_name.value}/"

    # load indexes
    indexes = load_indexes(dataset_name)
    assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    dataset = GraphsDataset(dataset_name)
    # add to config info about dataset
    train_config["net_params"]["max_wl_role_index"] = dataset.max_num_node
    train_config["net_params"]["num_classes"] = dataset.num_classes
    train_config["net_params"]["num_node_type"] = dataset.num_node_type
    train_config["net_params"]["num_edge_type"] = dataset.num_edge_type

    # prepare dataset
    time_measure(prepare_dataset, "gt", dataset_name, "preparation")(
        dataset, train_config
    )
    # prepare_dataset(dataset, train_config)

    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")
        test_idx = fold["test"]
        train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
        dataset.upload_indexes(train_idx, val_idx, test_idx)

        model, device = time_measure(
            train_graph_transformer, "gt", dataset_name, "training"
        )(dataset, train_config)

        eval_idx = list(range(128))
        dataset.upload_indexes(eval_idx, eval_idx, eval_idx)
        eval_loader = DataLoader(
            dataset.test,
            batch_size=config["params"]["batch_size"],
            shuffle=False,
            collate_fn=dataset.collate,
        )

        predictions = time_measure(get_prediction, "gt", dataset_name, "evaluation")(
            model, device, eval_loader
        )

        # scores_class, best_epoch_scores = train_graph_transformer(
        #     dataset, train_config
        # )
        break
    del dataset
    # evaluate model


if __name__ == "__main__":
    for dataset_name in DatasetName:
        print(f"Performing experiment for {dataset_name}")
        perform_experiment(dataset_name)
