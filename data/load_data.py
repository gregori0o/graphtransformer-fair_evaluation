from data.molecules import MoleculeDatasetIndexed
from utils import DATASET_NAME


def LoadData(dataset_name: DATASET_NAME, train_idx, val_idx, test_idx):
    # handling for (ZINC) molecule dataset
    if dataset_name == DATASET_NAME.ZINC:
        return MoleculeDatasetIndexed(train_idx, val_idx, test_idx)
