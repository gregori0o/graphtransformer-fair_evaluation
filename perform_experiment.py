import json
import time

import numpy as np
from sklearn.model_selection import train_test_split

from data.load_data import LoadData
from evaluation_config import K_FOLD, R_EVALUATION
from train_graph_transformer import train_graph_transformer
from utils import DATASET_NAME

configurations = {
    DATASET_NAME.ZINC: {
        "path": "molecules",
    }
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def perform_experiment(dataset_name):
    config = configurations.get(dataset_name)
    if config is None:
        raise ValueError(f"There is no configuration for {dataset_name} dataset.")

    # load indexes
    with open(f"data-splits/{config['path']}/data_split.json", "r") as f:
        indexes = json.load(f)
    assert len(indexes) == K_FOLD, "Re-generate splits for new K_FOLD."

    # loop over splits
    scores = []
    for i, fold in enumerate(indexes):
        print(f"FOLD {i}")

        # train_idx = fold["train"]
        test_idx = fold["test"]

        ## load datasets for this split (train and test)
        ## get best model for train data

        # evaluate model R times
        scores_r = 0
        for _ in range(R_EVALUATION):
            train_idx, val_idx = train_test_split(fold["train"], test_size=0.2)
            dataset = LoadData(dataset_name, train_idx, val_idx, test_idx)
            acc = train_graph_transformer(dataset, "zinc_config.json")
            scores_r += acc
        scores_r /= R_EVALUATION
        print(f"MEAN ACC = {scores_r} in FOLD {i}")
        scores.append(scores_r)

    # evaluate model
    mean = np.mean(scores)
    std = np.std(scores)

    print(f"Evaluation of model on {dataset_name}")
    print(f"Mean acc: {mean}")
    print(f"STD acc: {std}")

    with open("zinc_config.json", "r") as f:
        train_config = json.load(f)

    train_config["dataset_name"] = dataset_name.value
    train_config["mean_score"] = mean
    train_config["std_score"] = std
    dumped = json.dumps(train_config, cls=NpEncoder)
    with open(
        f"results/result_GT_{dataset_name.value}_{time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')}.json",
        "w",
    ) as f:
        f.write(dumped)


if __name__ == "__main__":
    perform_experiment(DATASET_NAME.ZINC)
