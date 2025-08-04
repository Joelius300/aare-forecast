import os
from typing import cast, Optional

from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from mlflow.entities import RunInfo

from aare.feature_identifiers import FeatureIdentifiers
from aare.storage.metadata import AareModel, get_mlflow_info, serialize_model_info, load_model_info
from aare.utils import MODEL_FOLDER


def _make_model_path(name: str, version: str):
    return MODEL_FOLDER / f"{name}-{version}.pkl"


def save_model(
    name: str,
    version: str,
    model: GlobalForecastingModel,
    features: FeatureIdentifiers,
    run_info: RunInfo,
    override=False,
):
    """Store a model with all necessary information including a metadata file."""
    model_path = _make_model_path(name, version)
    meta_path = model_path.with_suffix(".json")
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

    model.save(model_path, clean=True)
    serialize_model_info(meta_path, meta)


def load_model(
    meta_path: Optional[os.PathLike] = None, name: Optional[str] = None, version: Optional[str] = None
) -> tuple[AareModel, GlobalForecastingModel]:
    """Load a model from a specified path. For local dev, can also provide name and version."""
    if not meta_path:
        if not name or not version:
            raise ValueError("name and version must be provided if meta_path is not supplied")

        meta_path = _make_model_path(name, version).with_suffix(".json")

    meta: AareModel = load_model_info(meta_path)
    assert issubclass(meta["model_cls"], GlobalForecastingModel), "model_cls is not a GlobalForecastingModel"
    model = cast(GlobalForecastingModel, meta["model_cls"].load(meta["model_path"]))

    return meta, model
