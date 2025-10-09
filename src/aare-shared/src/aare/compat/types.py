from typing import TypedDict, NotRequired, Optional

from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer


class DataTransformers(TypedDict):
    """Typing for the data_transformers argument of the historical_forecast (backtest) function."""

    series: NotRequired[InvertibleDataTransformer | Pipeline]
    past_covariates: NotRequired[BaseDataTransformer | Pipeline]
    future_covariates: NotRequired[BaseDataTransformer | Pipeline]


ExtremeLags = tuple[
    Optional[int],  # min target lag,
    Optional[int],  # max target lag,
    Optional[int],  # min past covariate lag,
    Optional[int],  # max past covariate lag,
    Optional[int],  # min future covariate lag,
    Optional[int],  # max future covariate lag,
    int,  # output shift,
]
