import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, overload

import yaml
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

from aare_train.compat.types import DataTransformers
from aare_train.fetching.feature_identifiers import FeatureIdentifiers
from aare_train.params import read_params
from aare_train.storage.metadata import AareModel, get_mlflow_info, serialize_model_info, load_model_info
from aare_train.paths import MODELS_FOLDER

# Using with_suffix on paths with multipart suffixes will give you wrong paths because it only changes the last part.
META_SUFFIX = ".json"
MODEL_SUFFIX = ".model.pkl"
SCALER_SUFFIX = ".scaler.pkl"
PARAMS_SUFFIX = ".params.yaml"

logger = logging.getLogger(__name__)


def _make_model_base_path_relative(name: str, version: str | None):
    if not version:  # should only be used for baseline models
        return Path(f"./{name}")

    return Path(f"./{name}-{version}")


def _clean_model_params(params: dict[str, Any]):
    # for now only the 'model' entry is problematic. could also check for trivial types instead.
    return {key: value for key, value in params.items() if key != "model"}


def get_current_commit() -> str | None:
    try:
        from git import Repo

        repo = Repo(search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception as e:
        logger.warning("Could not get current commit of repo, using None", exc_info=e)
        return None


def model_exists(
    name: str,
    version: str,
) -> bool:
    """Returns whether the model folder exists."""
    base_path = _make_model_base_path_relative(name, version)
    model_folder = MODELS_FOLDER / base_path

    return model_folder.is_dir()


def save_model(
    name: str,
    version: str,
    model: GlobalForecastingModel,
    features: FeatureIdentifiers,
    scalers: DataTransformers,
    run_info: Any,  # no type -> no import of mlflow in aare-train, at least if possible. maybe skinny if forced.
    hparams: dict[str, Any],
    override: bool = False,
):
    """
    Store a model with all necessary information including a metadata file.
    Note, this will also store things like the current time, current commit and derived info from the run_info and
    hparams for reproducibility, so supply them. Will refuse to override unless version is 'dev' or override=True.
    """
    base_path = _make_model_base_path_relative(name, version)
    model_folder = MODELS_FOLDER / base_path

    if model_folder.is_dir() and not override and version != "dev":
        raise ValueError(f"There already is a model at {model_folder} but override is False and version != dev")

    model_folder.mkdir(exist_ok=True, parents=True)

    # redundancy in the folder path and filename, but just want to make sure it's always clear which model
    # and version the file is by just looking at the file (not inside, not its parent).
    # must keep base_path.suffix because versioned models already contains periods.
    meta_path = model_folder / base_path.with_suffix(base_path.suffix + META_SUFFIX)

    model_cls = type(model)
    assert issubclass(model_cls, GlobalForecastingModel), f"model_cls '{model_cls}' is not a GlobalForecastingModel"

    # store paths in json as relative so you don't have to update them from dev to prod
    model_path_rel = base_path.with_suffix(base_path.suffix + MODEL_SUFFIX)
    scalers_path_rel = base_path.with_suffix(base_path.suffix + SCALER_SUFFIX)
    params_path_rel = base_path.with_suffix(base_path.suffix + PARAMS_SUFFIX)

    meta: AareModel = {
        "name": name,
        "version": version,
        "model_cls": model_cls,
        "model_path": str(model_path_rel),
        "scalers_path": str(scalers_path_rel),
        "params_path": str(params_path_rel),
        "features": features,
        "hparams": hparams,
        "hparams_internal": _clean_model_params(model.model_params),
        "origin": {
            "mlflow": get_mlflow_info(run_info),
            # yes, this insane snippet is required to get a json-serializable timezone aware timestamp from ms epoch.
            "train_time": datetime.fromtimestamp(run_info.start_time / 1000).astimezone().isoformat(),
            "last_commit": get_current_commit(),
        },
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
    d = cast(dict[str, Any], meta)  # pyright: ignore[reportInvalidCast]
    d[key] = base_path / d[key]


def load_model_meta(
    meta_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    make_paths_absolute: bool = True,
) -> AareModel:
    """Load a model meta dict from a specified path. For local dev, can also provide name and version."""
    if not meta_path:
        if not name:
            raise ValueError("name must be provided if meta_path is not supplied")
        if not version:
            logger.warning("Loading model without version, should only be used for baseline models!")

        base_path = _make_model_base_path_relative(name, version)
        # same path during training
        meta_path = MODELS_FOLDER / base_path / base_path.with_suffix(base_path.suffix + META_SUFFIX)

    meta_path = Path(meta_path).absolute()  # turn it into an absolute path, should only be relative during dev tho
    if not meta_path.is_file() and meta_path.suffix == META_SUFFIX:
        raise ValueError(f"The meta file '{meta_path}' does not exist or isn't a valid meta file.")
    meta: AareModel = load_model_info(meta_path)

    if not make_paths_absolute:
        return meta

    # turn all the relative paths in the json into absolute paths for easier handling
    model_folder = meta_path.parent
    for key in meta.keys():
        if key.endswith("_path"):
            _make_abs_path(meta, model_folder, key)

    return meta


def load_model_from_meta(
    meta: AareModel,
) -> tuple[GlobalForecastingModel, DataTransformers | None]:
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


@overload
def load_model(
    meta_path: str | Path,
    *,
    name: Literal[None] = None,
    version: Literal[None] = None,
) -> tuple[AareModel, GlobalForecastingModel, DataTransformers | None]:
    pass


@overload
def load_model(
    meta_path: Literal[None] = None,
    *,
    name: str,
    version: str | None,
) -> tuple[AareModel, GlobalForecastingModel, DataTransformers | None]:
    pass


def load_model(
    meta_path: str | Path | None = None,
    *,
    name: str | None = None,
    version: str | None = None,
) -> tuple[AareModel, GlobalForecastingModel, DataTransformers | None]:
    """Load a model from a specified path. For local dev, can also provide name and version."""
    meta = load_model_meta(meta_path, name, version, make_paths_absolute=True)
    model, scalers = load_model_from_meta(meta)

    return meta, model, scalers
