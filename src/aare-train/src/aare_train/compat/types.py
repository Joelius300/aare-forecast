from typing import TypedDict, NotRequired

from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import BaseDataTransformer, InvertibleDataTransformer


class DataTransformers(TypedDict):
    """Typing for the data_transformers argument of the historical_forecast (backtest) function."""

    series: NotRequired[InvertibleDataTransformer | Pipeline]
    past_covariates: NotRequired[BaseDataTransformer | Pipeline]
    future_covariates: NotRequired[BaseDataTransformer | Pipeline]


ExtremeLags = tuple[
    int | None,  # min target lag,
    int | None,  # max target lag,
    int | None,  # min past covariate lag,
    int | None,  # max past covariate lag,
    int | None,  # min future covariate lag,
    int | None,  # max future covariate lag,
    int,  # output shift,
]
