import itertools
import json
import os
import time

import numpy as np
import optuna
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split

from data.load_data import DatasetName, GraphsDataset, load_indexes
from evaluation_config import K_FOLD, R_EVALUATION
from train_graph_transformer import train_graph_transformer
from utils import NpEncoder

experiment_name = time.strftime("%Y_%m_%d_%Hh%Mm%Ss")


def prepare_dataset(dataset, train_config):
    if train_config["net_params"]["lap_pos_enc"]:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(
            train_config["net_params"]["pos_enc_dim"]
        )
        print("Time LapPE:", time.time() - st)

    if train_config["net_params"]["wl_pos_enc"]:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        print("Time WL PE:", time.time() - st)

    if train_config["net_params"]["full_graph"]:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        print("Time taken to convert to full graphs:", time.time() - st)

    if train_config["net_params"]["self_loop"]:
        st = time.time()
        print("[!] Converting the given graphs, adding self loops..")
        dataset._add_self_loops()
        print("Time taken to add self loops:", time.time() - st)


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
    prepare_dataset(dataset, train_config)

    # loop over splits
    scores = {
        "accuracy": [],
        "f1": [],
        "macro f1": [],
        "precision": [],
        "recall": [],
        "roc": [],
    }
    epoch_scores = {
        "accuracy": [],
        "f1": [],
        "macro f1": [],
        "precision": [],
        "recall": [],
        "roc": [],
    }
    best_epochs = []

    tuning_result = {}
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")
        needs_new_dataset = False

        ## get best model for train data
        if config.get("tune_hyperparameters"):
            needs_new_dataset = True
            best_params, best_acc = find_best_params(
                train_config, dataset, dataset_name, fold
            )

            train_config["params"].update(best_params["params"])
            train_config["net_params"].update(best_params["net_params"])
            tuning_result[i] = {
                "params": best_params["params"],
                "net_params": best_params["net_params"],
                "accuracy": best_acc,
            }

        if needs_new_dataset:
            dataset = GraphsDataset(dataset_name)
            prepare_dataset(dataset, train_config)
        # evaluate model R times
        scores_r = {
            "accuracy": 0,
            "f1": 0,
            "macro f1": 0,
            "precision": 0,
            "recall": 0,
            "roc": 0,
        }
        epoch_scores_r = {
            "accuracy": 0,
            "f1": 0,
            "macro f1": 0,
            "precision": 0,
            "recall": 0,
            "roc": 0,
        }
        test_idx = fold["test"]
        for _ in range(R_EVALUATION):
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.1)
            dataset.upload_indexes(train_idx, val_idx, test_idx)
            scores_class, best_epoch_scores = train_graph_transformer(
                dataset, train_config
            )
            for key in scores_r.keys():
                scores_r[key] += scores_class[key]
            for key in epoch_scores_r.keys():
                epoch_scores_r[key] += best_epoch_scores[key]
            best_epochs.append(best_epoch_scores["epoch"])
        for key in scores_r.keys():
            scores_r[key] /= R_EVALUATION
            epoch_scores_r[key] /= R_EVALUATION
        print(f"MEAN SCORES = {scores_r} in FOLD {i}")
        for key in scores_r.keys():
            scores[key].append(scores_r[key])
            epoch_scores[key].append(epoch_scores_r[key])

    del dataset
    # evaluate model
    summ = {}
    for key in scores.keys():
        summ[key] = {}
        summ[key]["mean"] = np.mean(scores[key])
        summ[key]["std"] = np.std(scores[key])

    epoch_summ = {}
    for key in epoch_scores.keys():
        epoch_summ[key] = {}
        epoch_summ[key]["mean"] = np.mean(epoch_scores[key])
        epoch_summ[key]["std"] = np.std(epoch_scores[key])

    # scores are acc, precision, recall, F1 and ROC
    print(f"Evaluation of model on {dataset_name}")
    print(f"Scores: {scores}")
    print(f"Summary: {summ}")

    train_config["dataset_name"] = dataset_name.value
    train_config["run_config"] = config
    train_config["tune_hyperparameters"] = tuning_result
    train_config["summary_scores"] = summ
    train_config["scores"] = scores
    train_config["ES scores"] = epoch_scores
    train_config["ES summary_scores"] = epoch_summ
    train_config["ES best_epochs"] = best_epochs
    train_config["r_evaluation"] = R_EVALUATION
    train_config["k_fold"] = K_FOLD
    del train_config["net_params"]["device"]
    dumped = json.dumps(train_config, cls=NpEncoder)
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
    with open(
        f"results/{experiment_name}/result_GT_{dataset_name.value}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}.json",
        "w",
    ) as f:
        f.write(dumped)


if __name__ == "__main__":
    perform_experiment(DatasetName.DD)
    perform_experiment(DatasetName.NCI1)
    perform_experiment(DatasetName.ENZYMES)
    perform_experiment(DatasetName.PROTEINS)
    perform_experiment(DatasetName.IMDB_BINARY)
    perform_experiment(DatasetName.IMDB_MULTI)
    perform_experiment(DatasetName.REDDIT_BINARY)
    perform_experiment(DatasetName.REDDIT_MULTI)
    perform_experiment(DatasetName.COLLAB)
    # perform_experiment(DatasetName.MOLHIV)
