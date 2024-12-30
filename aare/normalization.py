import pickle
from os import PathLike

from sklearn.preprocessing import StandardScaler

from aare.utils import DATA_FOLDER

SCALER_PATH = DATA_FOLDER / "scaler.pkl"


def load_scaler(path: PathLike = SCALER_PATH):
    with open(path, "rb") as file:
        return pickle.load(file)


def store_scaler(scaler: StandardScaler, path: PathLike = SCALER_PATH):
    # could also the text-based version from AICH/normalization.py
    with open(path, "wb") as file:
        pickle.dump(scaler, file)
