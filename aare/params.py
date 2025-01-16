from typing import TypedDict, cast

import dvc.api


class GeneralParams(TypedDict):
    frequency: str  # this is used everywhere and integrated so tightly, that it cannot simply be changed
    forecast_horizon: int


class CleanupParams(TypedDict):
    low_cutoff: float
    high_cutoff: float
    diff_threshold: float


class InterpolateParams(TypedDict):
    linear_gap_bound: int
    cubic_gap_bound: int


class SplitParams(TypedDict):
    # these are lower bounds for the respective splits ( train_period = [train_split; val_split[ )
    train_split: str
    val_split: str
    test_split: str  # test data has no upper bound


class ValidationParams(TypedDict):
    stride: int
    min_lookback_hours: int


class Params(TypedDict):
    general: GeneralParams
    cleanup: CleanupParams
    interpolate: InterpolateParams
    split: SplitParams
    validation: ValidationParams


def read_params() -> Params:
    """Returns the dvc params with appropriate typing (dvc.api.params_show)."""
    # This function should be extended when more functionality from params_show is needed.

    return cast(Params, dvc.api.params_show())
