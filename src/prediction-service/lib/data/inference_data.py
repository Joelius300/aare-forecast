from typing import TypedDict, NotRequired, Optional

from darts import TimeSeries


class InferenceData(TypedDict):
    # no batch support (or necessity)
    # not provided is the same as None here, that's not always the case (!)
    series: TimeSeries  # some models allow covariate-only pred, but we'll never use it so non-nullable
    past_covariates: NotRequired[Optional[TimeSeries]]
    future_covariates: NotRequired[Optional[TimeSeries]]
