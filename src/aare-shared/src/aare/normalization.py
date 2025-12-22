import pickle
from os import PathLike

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

from aare.compat.types import DataTransformers
from aare.paths import DATA_FOLDER

SCALER_PATH = DATA_FOLDER / "scaler.pkl"


def load_scaler(path: PathLike = SCALER_PATH):
    with open(path, "rb") as file:
        return pickle.load(file)


def store_scaler(scaler: StandardScaler, path: PathLike = SCALER_PATH):
    # could also the text-based version from AICH/normalization.py
    with open(path, "wb") as file:
        pickle.dump(scaler, file)


def get_scalers(
    train_target_subs: list[TimeSeries],
    *,
    train_pc_subs: list[TimeSeries] | None = None,
    train_fc_subs: list[TimeSeries] | None = None,
) -> DataTransformers:
    """Train darts compatible StandardScalers for target, past cov and future cov."""
    scaler_target = Scaler(StandardScaler(), global_fit=True)
    scaler_pc = Scaler(StandardScaler(), global_fit=True) if train_pc_subs is not None else None
    scaler_fc = Scaler(StandardScaler(), global_fit=True) if train_fc_subs is not None else None

    scaler_target.fit(train_target_subs)

    # darts can't handle if the scaler is just None, it must not be present in the dict...
    dt: DataTransformers = {
        "series": scaler_target,
    }

    if scaler_pc:
        assert train_pc_subs is not None
        scaler_pc.fit(train_pc_subs)
        dt.update(past_covariates=scaler_pc)
    if scaler_fc:
        assert train_fc_subs is not None
        scaler_fc.fit(train_fc_subs)
        dt.update(future_covariates=scaler_fc)

    return dt
