from typing import TypedDict, Literal, Any, NotRequired


class GeneralParams(TypedDict):
    frequency: str  # this is used everywhere and integrated so tightly, that it cannot simply be changed
    forecast_horizon: int


class OutliersParams(TypedDict):
    low_cutoff: float
    high_cutoff: float
    diff_threshold: float


class InterpolateParams(TypedDict):
    linear_gap_bound: NotRequired[int]
    cubic_gap_bound: NotRequired[int]
    median_gap_bound: NotRequired[int]


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


class FeatureParams(TypedDict):
    outliers: NotRequired[OutliersParams]
    interpolate: NotRequired[InterpolateParams]
    custom: NotRequired[dict[str, Any]]  # for feature-specific custom configuration


FeaturesParams = dict[str, FeatureParams]


class Params(TypedDict):
    general: GeneralParams
    features: FeaturesParams
    split: SplitParams
    validation: ValidationParams
    timesfm: TimesfmParams
