"""
OUTDATED: We want to use the darts scaler and need a flexible way to handle a) different time spans / splits and b)
different feature sets, so retraining the scaler just before training might be better anyway.
"""

import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from aare.fetching.AareDataset import AareDataset
from aare.constants import TEMP
from aare.normalization import store_scaler
from aare.preparation import resample, remove_faulty_periods_aare_temp, remove_outliers_aare_temp


logger = logging.getLogger(__name__)


def prepare_data(dataset: AareDataset):
    train = dataset.get_train()

    train = resample(train)
    train = remove_faulty_periods_aare_temp(train)
    train = remove_outliers_aare_temp(train)

    return train


def train_scaler(train: pd.DataFrame):
    scaler = StandardScaler()
    X = train[TEMP].to_numpy()
    X = X.reshape(-1, 1)

    scaler.fit(X)

    return scaler


def main():
    dataset = AareDataset.from_conf()

    train = prepare_data(dataset)
    scaler = train_scaler(train)
    store_scaler(scaler)

    logger.info(f"Trained StandardScaler on {len(train)} rows.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
