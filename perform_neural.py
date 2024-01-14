import itertools
import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

from data.load_data import DatasetName, GraphsDataset, load_indexes
from train_graph_transformer import train_graph_transformer
from utils import NpEncoder

R_EVALUATION = 4


def get_all_params(param_grid):
    return [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]


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


def perform_experiment(dataset_name):
    config = {
        "config_path": "configs/zinc_config.json",
        "net_params_grid": {
            "in_feat_dropout": [0, 0.15, 0.3],
            "dropout": [0, 0.15, 0.3],
            # "L": [6, 10, 14],
            # "n_heads": [4, 8, 12],
            # "hidden_dim": [64, 128],
            # "out_dim": [64, 128],
        },
        "params_grid": {
            "init_lr": [7e-4, 7e-5, 7e-6],
            "lr_reduce_factor": [0.3, 0.5, 0.8],
        },
        "tune_hyperparameters": True,
    }

    with open(config["config_path"], "r") as f:
        train_config = json.load(f)
    train_config["out_dir"] = f"out/{dataset_name.value}/"

    # # load indexes
    # indexes = load_indexes(dataset_name)
    # assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    dataset = GraphsDataset(dataset_name)
    # add to config info about dataset
    train_config["net_params"]["max_wl_role_index"] = dataset.max_num_node
    train_config["net_params"]["num_classes"] = dataset.num_classes
    train_config["net_params"]["num_node_type"] = dataset.num_node_type
    train_config["net_params"]["num_edge_type"] = dataset.num_edge_type

    scores = []

    train_tmp, test_indices = train_test_split(
        list(range(len(dataset.graphs))),
        test_size=0.2,
        stratify=dataset.labels,
        random_state=123,
        shuffle=True,
    )
    train_labels = [l for i, l in enumerate(dataset.labels) if i in train_tmp]
    train_tmp = np.sort(train_tmp)
    train_indices, val_indices = train_test_split(
        train_tmp,
        test_size=0.125,
        stratify=train_labels,
        random_state=123,
        shuffle=True,
    )

    print("train_indices", len(train_indices))
    print("val_indices", len(val_indices))
    print("test_indices", len(test_indices))

    # prepare dataset
    prepare_dataset(dataset, train_config)

    tuning_result = {}
    print("Start experiment")
    needs_new_dataset = False

    ## get best model for train data
    if config.get("tune_hyperparameters"):
        if (
            "lap_pos_enc" in config["net_params_grid"]
            or "wl_pos_enc" in config["net_params_grid"]
            or "full_graph" in config["net_params_grid"]
            or "self_loop" in config["net_params_grid"]
        ):
            needs_new_dataset = True
        best_acc = 0
        best_params = ({}, {})
        params = get_all_params(config["params_grid"])
        net_params = get_all_params(config["net_params_grid"])
        for param in params:
            train_config["params"].update(param)
            for net_param in net_params:
                train_config["net_params"].update(net_param)
                if (
                    train_config["net_params"]["lap_pos_enc"]
                    == train_config["net_params"]["wl_pos_enc"]
                ):
                    continue
                if needs_new_dataset:
                    dataset = GraphsDataset(dataset_name)
                    prepare_dataset(dataset, train_config)
                dataset.upload_indexes(
                    train_indices, val_indices, val_indices
                )  # test_idx <- val_idx
                acc = train_graph_transformer(dataset, train_config)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (param.copy(), net_param.copy())

        train_config["params"].update(best_params[0])
        train_config["net_params"].update(best_params[1])
        tuning_result[0] = {"params": best_params[0], "net_params": best_params[1]}

    if needs_new_dataset:
        dataset = GraphsDataset(dataset_name)
        prepare_dataset(dataset, train_config)
    # evaluate model R times
    for _ in range(R_EVALUATION):
        train_config["params"]["seed"] += 1
        dataset.upload_indexes(train_indices, val_indices, test_indices)
        acc = train_graph_transformer(dataset, train_config)
        scores.append(acc)

    del dataset
    # evaluate model
    mean = np.mean(scores)
    std = np.std(scores)

    # score is MAE for regression and ACC for other
    print(f"Evaluation of model on {dataset_name}")
    print(f"Mean score: {mean}")
    print(f"STD score: {std}")

    train_config["dataset_name"] = dataset_name.value
    train_config["run_config"] = config
    train_config["tune_hyperparameters"] = tuning_result
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
    perform_experiment(DatasetName.NEURAL)
