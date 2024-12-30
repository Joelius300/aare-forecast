import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from aare.AareDataset import AareDataset
from aare.constants import TEMP
from aare.normalization import store_scaler
from aare.params import read_params
from aare.preparation import resample, remove_faulty_periods, remove_outliers
from aare.remote_existenz_store import RemoteExistenzStore


logger = logging.getLogger(__name__)


def prepare_data(dataset: AareDataset):
    train = dataset.get_train()

    train = resample(train)
    train = remove_faulty_periods(train)
    train = remove_outliers(train)

    return train


def train_scaler(train: pd.DataFrame):
    scaler = StandardScaler()
    X = train[TEMP].to_numpy()
    X = X.reshape(-1, 1)

    scaler.fit(X)

    return scaler


def main():
    params = read_params()["training"]
    store = RemoteExistenzStore()
    dataset = AareDataset(store, params["val_split"], params["test_split"])

    train = prepare_data(dataset)
    scaler = train_scaler(train)
    store_scaler(scaler)

    logger.info(f"Trained StandardScaler on {len(train)} rows.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
