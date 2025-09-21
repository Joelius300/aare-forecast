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
from aare.utils import MODELS_FOLDER

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


def load_model_meta(
    meta_path: Optional[os.PathLike | str] = None, name: Optional[str] = None, version: Optional[str] = None
) -> tuple[AareModel, Path]:
    if not meta_path:
        if not name or not version:
            raise ValueError("name and version must be provided if meta_path is not supplied")

        base_path = _make_model_base_path_relative(name, version)
        # same path during training
        meta_path = MODELS_FOLDER / base_path / base_path.with_suffix(META_SUFFIX)

    meta_path = Path(meta_path)
    if not meta_path.is_file() and meta_path.suffix == META_SUFFIX:
        raise ValueError(f"The meta file '{meta_path}' does not exist or isn't a valid meta file.")
    meta: AareModel = load_model_info(meta_path)

    base_path = meta_path.parent
    return meta, base_path


def load_model(
    meta: AareModel,
    meta_path_or_base: os.PathLike | str | Path,
) -> tuple[GlobalForecastingModel, Optional[DataTransformers]]:
    """Load a model from a specified path. For local dev, can also provide name and version."""
    # TODO instead of requiring the base path to be passed here, could also make the paths in the meta dict absolute
    # when reading them in (because then you know where the meta file lies).
    model_cls = meta["model_cls"]
    assert issubclass(model_cls, GlobalForecastingModel), f"model_cls '{model_cls}' is not a GlobalForecastingModel"

    meta_path_or_base = Path(meta_path_or_base)
    if meta_path_or_base.is_file() and meta_path_or_base.suffix == META_SUFFIX:
        # the meta file path was passed, use its parent as base
        model_base_path = meta_path_or_base.parent
    elif meta_path_or_base.is_dir():
        # base path was passed, use it directly
        model_base_path = meta_path_or_base
    else:
        raise ValueError(
            f"The passed path '{meta_path_or_base}' doesn't point to the meta file or the model base path."
        )

    model_path = model_base_path / meta["model_path"]
    scalers_path = model_base_path / meta["scalers_path"]

    model = cast(GlobalForecastingModel, model_cls.load(model_path))

    with open(scalers_path, "rb") as file:
        scalers = pickle.load(file)

    return model, scalers
