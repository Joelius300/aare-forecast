import logging
import json
import importlib
import os
from pathlib import Path, PosixPath, WindowsPath
from typing import TypedDict, cast

from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from mlflow.entities import RunInfo

from aare.feature_identifiers import FeatureIdentifiers

logger = logging.getLogger(__name__)


class MlFlowInfo(TypedDict):
    run_name: str
    exp_id: str
    run_id: str


def get_mlflow_info(run_info: RunInfo) -> MlFlowInfo:
    return {
        "run_name": str(run_info.run_name),
        "exp_id": run_info.experiment_id,
        "run_id": run_info.run_id,
    }


class AareModel(TypedDict):
    """Info about an aare model, needed for inference."""

    model_path: str
    model_cls: type[GlobalForecastingModel]
    features: FeatureIdentifiers
    # metadata for transparency and diagnostics
    name: str
    version: str
    mlflow: MlFlowInfo  # for precise traceability/transparency


def _get_class_by_name(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class _ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return o.__module__ + "." + o.__qualname__
        elif isinstance(o, (Path, PosixPath, WindowsPath)):
            return str(o)

        return super().default(o)


def _class_decoder(obj):
    # treat all keys with *_cls as type and import the referenced class
    for key, value in obj.items():
        if key.endswith("_cls") and isinstance(value, str):
            try:
                cls = _get_class_by_name(value)
                obj[key] = cls
                logger.debug(f"Successfully imported '{value}': {cls}")
            except (ImportError, AttributeError):
                # leave it as string if class can't be imported but print warning
                logger.warning(f"Could not import class '{value}' stored in property '{key}'")

    return obj


def serialize_model_info(path: os.PathLike, model: AareModel, indent=2) -> None:
    """Write a model info dict to a json file."""
    with open(str(path), "wt") as file:
        json.dump(model, file, cls=_ClassEncoder, indent=indent)


def load_model_info(path: os.PathLike | str) -> AareModel:
    """Read a model info dict from json and import specified model type."""
    with open(str(path), "rt") as file:
        info = json.load(file, object_hook=_class_decoder)

    return cast(AareModel, info)
