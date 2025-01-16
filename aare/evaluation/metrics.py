from typing import TypedDict, cast

from darts import TimeSeries
from darts.metrics import mae, rmse


class Metrics(TypedDict):
    mae: float
    rmse: float


def calc_metrics(actual: TimeSeries, prediction: TimeSeries):
    """Returns a fully populated instance of Metrics."""
    to_eval = dict(mae=mae, rmse=rmse)

    metrics = {key: cast(float, metric(actual, prediction)) for (key, metric) in to_eval.items()}

    return cast(Metrics, metrics)
