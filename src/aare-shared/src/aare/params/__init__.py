from pathlib import Path

from aare.params.params_manager import params_manager_singleton
from aare.params.params_types import (
    Params,
    GeneralParams,
    FeatureParams,
    FeaturesParams,
    InterpolateParams,
    OutliersParams,
    SplitParams,
    TimesfmParams,
    ValidationParams,
)


def read_params(*, ensure_dvc=False) -> Params:
    """
    Returns the params with appropriate typing from the configured source.
    Set ensure_dvc to true to raise an error if not running in a DVC context.
    """
    return params_manager_singleton.read_params(ensure_dvc=ensure_dvc)


def set_params_file(params_file_path: str | Path):
    """
    Sets the path of the params file for the application.
    Can only be set once and will be used from then on by "read_params"!
    """
    params_manager_singleton.set_params_file(params_file_path)


__all__ = [
    "read_params",
    "set_params_file",
    "Params",
    "GeneralParams",
    "FeatureParams",
    "FeaturesParams",
    "InterpolateParams",
    "OutliersParams",
    "SplitParams",
    "TimesfmParams",
    "ValidationParams",
]
