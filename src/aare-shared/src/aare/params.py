from typing import TypedDict, cast, Literal

from aare.utils import PROJECT_ROOT


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


class TimesfmParams(TypedDict):
    version: Literal["200m", "500m"]


class Params(TypedDict):
    general: GeneralParams
    cleanup: CleanupParams
    interpolate: InterpolateParams
    split: SplitParams
    validation: ValidationParams
    timesfm: TimesfmParams


def read_params() -> Params:
    """
    Returns the dvc params with appropriate typing (dvc.api.params_show).

    If not in a dvc context/project, "params.yaml" is attempted to be read.
    """
    # This function should be extended when more functionality from params_show is needed.
    # TODO the params that the model were trained with must be bundled with the model!

    params = None
    if (PROJECT_ROOT / ".dvc").is_dir():
        import dvc.api

        params = dvc.api.params_show()
    else:
        import yaml

        with open(PROJECT_ROOT / "params.yaml", "rt") as file:
            params = yaml.safe_load(file)

    assert params is not None, "params is None after reading"

    return cast(Params, params)
