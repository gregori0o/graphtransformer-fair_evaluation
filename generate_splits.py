import json

import numpy as np
import pandas as pd
from dgl.data import TUDataset

from data.load_data import DATA_SPLITS_DIR, DatasetName
from evaluation_config import K_FOLD


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def generate(dataset_name: DatasetName):
    np.random.seed(12)
    path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}.json"

    tmp_dataset = TUDataset(dataset_name.value, f"/tmp/{dataset_name.value}")
    size = len(tmp_dataset)

    indexes = np.random.permutation(size)
    fold_size = size // K_FOLD
    folds = []
    for i in range(K_FOLD):
        test_fold = indexes[fold_size * i : fold_size * (i + 1)]
        train_fold = np.r_[indexes[: fold_size * i], indexes[fold_size * (i + 1) :]]
        folds.append({"train": train_fold, "test": test_fold})
    dumped = json.dumps(folds, cls=NpEncoder)
    with open(path, "w") as f:
        f.write(dumped)


if __name__ == "__main__":
    for dataset_name in DatasetName:
        print(type(dataset_name), dataset_name.value)
        generate(dataset_name)
