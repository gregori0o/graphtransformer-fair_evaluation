import json

import numpy as np
import pandas as pd

from evaluation_config import K_FOLD
from utils import DATASET_NAME

configurations = {
    DATASET_NAME.ZINC: {"path": "molecules", "size": 1000, "from_file": True}
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


def generate(dataset_name: DATASET_NAME):
    np.random.seed(12)
    config = configurations.get(dataset_name)
    if config is None:
        raise ValueError(f"There is no configuration for {dataset_name} dataset.")
    path = config["path"]
    from_file = config.get("from_file")
    if from_file:
        index_path = f"./data/{path}/"
        train = pd.read_csv(index_path + "train.index", header=None).values.reshape(-1)
        test = pd.read_csv(index_path + "test.index", header=None).values.reshape(-1)
        val = pd.read_csv(index_path + "val.index", header=None).values.reshape(-1)
        indexes = np.random.permutation(np.r_[train, test, val])
        size = indexes.shape[0]
    else:
        size = config["size"]
        indexes = np.random.permutation(size)
    fold_size = size // K_FOLD
    folds = []
    for i in range(K_FOLD):
        test_fold = indexes[fold_size * i : fold_size * (i + 1)]
        train_fold = np.r_[indexes[: fold_size * i], indexes[fold_size * (i + 1) :]]
        folds.append({"train": train_fold, "test": test_fold})
    dumped = json.dumps(folds, cls=NpEncoder)
    with open(f"./data-splits/{path}/data_split.json", "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    generate(DATASET_NAME.ZINC)
