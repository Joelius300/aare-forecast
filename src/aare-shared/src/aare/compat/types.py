from typing import TypedDict, NotRequired

from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import BaseDataTransformer


class DataTransformers(TypedDict):
    """Typing for the data_transformers argument of the historical_forecast (backtest) function."""

    series: NotRequired[BaseDataTransformer | Pipeline]
    past_covariates: NotRequired[BaseDataTransformer | Pipeline]
    future_covariates: NotRequired[BaseDataTransformer | Pipeline]
