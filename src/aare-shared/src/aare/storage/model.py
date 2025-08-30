import os
import pickle
from pathlib import Path
from typing import cast, Optional

from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from mlflow.entities import RunInfo

from aare.compat.types import DataTransformers
from aare.feature_identifiers import FeatureIdentifiers
from aare.storage.metadata import AareModel, get_mlflow_info, serialize_model_info, load_model_info
from aare.utils import MODEL_FOLDER

# META_SUFFIX only has one part because with_suffix only changes the last, so we can use it as root/anchor.
# Using with_suffix on model or scaler path will give you wrong paths because the .model / .scaler stays.
META_SUFFIX = ".json"
MODEL_SUFFIX = ".model.pkl"
SCALER_SUFFIX = ".scaler.pkl"


def _make_meta_path(name: str, version: str):
    return MODEL_FOLDER / f"{name}-{version}{META_SUFFIX}"


def save_model(
    name: str,
    version: str,
    model: GlobalForecastingModel,
    features: FeatureIdentifiers,
    scalers: DataTransformers,
    run_info: RunInfo,
    override=False,
):
    """Store a model with all necessary information including a metadata file."""
    meta_path = _make_meta_path(name, version)
    model_path = meta_path.with_suffix(MODEL_SUFFIX)

    if meta_path.exists() and not override and version != "dev":
        raise ValueError(f"There already is a model at {meta_path} but override is False")

    meta: AareModel = {
        "model_path": str(model_path),
        "model_cls": type(model),
        "features": features,
        "name": name,
        "version": version,
        "mlflow": get_mlflow_info(run_info),
    }

    assert issubclass(meta["model_cls"], GlobalForecastingModel), "model_cls is not a GlobalForecastingModel"

    serialize_model_info(meta_path, meta)
    model.save(model_path, clean=True)
    with open(meta_path.with_suffix(SCALER_SUFFIX), "wb") as file:
        pickle.dump(scalers, file)


def load_model(
    meta_path: Optional[os.PathLike | str] = None, name: Optional[str] = None, version: Optional[str] = None
) -> tuple[AareModel, GlobalForecastingModel, Optional[DataTransformers]]:
    """Load a model from a specified path. For local dev, can also provide name and version."""
    if not meta_path:
        if not name or not version:
            raise ValueError("name and version must be provided if meta_path is not supplied")

        meta_path = _make_meta_path(name, version)

    meta: AareModel = load_model_info(meta_path)
    assert issubclass(meta["model_cls"], GlobalForecastingModel), "model_cls is not a GlobalForecastingModel"
    model = cast(GlobalForecastingModel, meta["model_cls"].load(meta["model_path"]))

    with open(Path(meta_path).with_suffix(SCALER_SUFFIX), "rb") as file:
        scalers = pickle.load(file)

    return meta, model, scalers
