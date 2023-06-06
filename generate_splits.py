import json

import numpy as np
import pandas as pd
from dgl.data import TUDataset

from data.load_data import DATA_SPLITS_DIR, DatasetName
from evaluation_config import K_FOLD
from utils import NpEncoder


def generate(dataset_name: DatasetName, limit=None):
    np.random.seed(12)
    path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}.json"

    tmp_dataset = TUDataset(dataset_name.value, raw_dir=f"/tmp/{dataset_name.value}")
    size = len(tmp_dataset)

    indexes = np.random.permutation(size)
    if limit is not None:
        size = min(limit, size)
        indexes = indexes[:size]
        path = f"data/{DATA_SPLITS_DIR}/{dataset_name.value}_{limit}.json"
    if K_FOLD == 1:
        train_size = int(0.8 * size)
        folds = [{"train": indexes[:train_size], "test": indexes[train_size:]}]
        dumped = json.dumps(folds, cls=NpEncoder)
        with open(path, "w") as f:
            f.write(dumped)
        return
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
        generate(dataset_name)
