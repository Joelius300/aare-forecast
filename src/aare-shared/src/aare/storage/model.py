import os
import pickle
from pathlib import Path
from typing import cast, Optional

import yaml
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare.compat.types import DataTransformers
from aare.feature_identifiers import FeatureIdentifiers
from aare.params import read_params
from aare.storage.metadata import AareModel, get_mlflow_info, serialize_model_info, load_model_info
from aare.paths import MODELS_FOLDER

# Using with_suffix on paths with multipart suffixes will give you wrong paths because it only changes the last part.
META_SUFFIX = ".json"
MODEL_SUFFIX = ".model.pkl"
SCALER_SUFFIX = ".scaler.pkl"
PARAMS_SUFFIX = ".params.yaml"


def _make_model_base_path_relative(name: str, version: str):
    return Path(f"./{name}-{version}")


def save_model(
    name: str,
    version: str,
    model: GlobalForecastingModel,
    features: FeatureIdentifiers,
    scalers: DataTransformers,
    run_info,  # no type -> no import of mlflow in aare-shared, at least if possible. maybe skinny if forced.
    override=False,
):
    """Store a model with all necessary information including a metadata file."""
    base_path = _make_model_base_path_relative(name, version)
    model_folder = MODELS_FOLDER / base_path
    model_folder.mkdir(exist_ok=True, parents=True)

    # redundancy in the folder path and filename, but just want to make sure it's always clear which model
    # and version the file is by just looking at the file (not inside, not its parent).
    meta_path = model_folder / base_path.with_suffix(META_SUFFIX)

    if meta_path.exists() and not override and version != "dev":
        raise ValueError(f"There already is a model at {meta_path} but override is False")

    model_cls = type(model)
    assert issubclass(model_cls, GlobalForecastingModel), f"model_cls '{model_cls}' is not a GlobalForecastingModel"

    # store paths in json as relative so you don't have to update them from dev to prod
    model_path_rel = base_path.with_suffix(MODEL_SUFFIX)
    scalers_path_rel = base_path.with_suffix(SCALER_SUFFIX)
    params_path_rel = base_path.with_suffix(PARAMS_SUFFIX)

    meta: AareModel = {
        "model_path": str(model_path_rel),
        "scalers_path": str(scalers_path_rel),
        "params_path": str(params_path_rel),
        "model_cls": model_cls,
        "features": features,
        "name": name,
        "version": version,
        "mlflow": get_mlflow_info(run_info),
    }

    # save model (meta) info (json)
    serialize_model_info(meta_path, meta)
    # save params(.yaml) with current values
    params = read_params(ensure_dvc=True)
    with open(model_folder / params_path_rel, "wt") as params_file:
        yaml.dump(params, params_file)
    # save model as pickle (via darts) -> might save multiple files
    model.save(model_folder / model_path_rel, clean=True)
    # save scalers as pickle
    with open(model_folder / scalers_path_rel, "wb") as file:
        pickle.dump(scalers, file)


def _make_abs_path(meta: AareModel, base_path: Path, key: str):
    d = cast(dict, meta)
    d[key] = base_path / d[key]
    meta = cast(AareModel, d)

    return meta


def load_model_meta(
    meta_path: Optional[os.PathLike | str | Path] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    make_paths_absolute=True,
):
    """Load a model meta dict from a specified path. For local dev, can also provide name and version."""
    if not meta_path:
        if not name or not version:
            raise ValueError("name and version must be provided if meta_path is not supplied")

        base_path = _make_model_base_path_relative(name, version)
        # same path during training
        meta_path = MODELS_FOLDER / base_path / base_path.with_suffix(META_SUFFIX)

    meta_path = Path(meta_path).absolute()  # turn it into an absolute path, should only be relative during dev tho
    if not meta_path.is_file() and meta_path.suffix == META_SUFFIX:
        raise ValueError(f"The meta file '{meta_path}' does not exist or isn't a valid meta file.")
    meta: AareModel = load_model_info(meta_path)

    if not make_paths_absolute:
        return meta

    # turn all the relative paths in the json into absolute paths for easier handling
    model_folder = meta_path.parent
    for key in meta.keys():
        key: str
        if key.endswith("_path"):
            _make_abs_path(meta, model_folder, key)

    return meta


def load_model_from_meta(
    meta: AareModel,
) -> tuple[GlobalForecastingModel, Optional[DataTransformers]]:
    """Load a model and its scalers from a meta dict."""
    model_cls = meta["model_cls"]
    assert issubclass(model_cls, GlobalForecastingModel), f"model_cls '{model_cls}' is not a GlobalForecastingModel"

    model_path = Path(meta["model_path"])
    assert model_path.is_absolute()
    scalers_path = Path(meta["scalers_path"])
    assert scalers_path.is_absolute()

    model = cast(GlobalForecastingModel, model_cls.load(model_path))

    with open(scalers_path, "rb") as file:
        scalers = pickle.load(file)

    return model, scalers


def load_model(
    meta_path: Optional[os.PathLike | str | Path] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
) -> tuple[AareModel, GlobalForecastingModel, Optional[DataTransformers]]:
    """Load a model from a specified path. For local dev, can also provide name and version."""
    meta = load_model_meta(meta_path, name, version, make_paths_absolute=True)
    model, scalers = load_model_from_meta(meta)

    return meta, model, scalers
