from typing import cast, TypedDict, Optional

from darts import TimeSeries
from pandas import Timedelta, Timestamp

from aare.evaluation.metrics import Metrics, calc_metrics


class LastPrediction(TypedDict):
    actual: TimeSeries
    prediction: TimeSeries
    lookback: Timedelta
    metrics: Optional[Metrics]


def get_last_prediction(
    all_data: TimeSeries,
    last_forecast: TimeSeries,
    lookback_hours: int,
    add_metrics=True,
):
    """Combine the actual and prediction series with a lookback period and optionally metrics."""
    assert lookback_hours >= 0, "lookback_hours must be positive"
    lookback = cast(Timedelta, Timedelta(hours=lookback_hours))  # cannot be NaT
    pred_start = cast(Timestamp, last_forecast.start_time())
    actual = all_data[pred_start - lookback : last_forecast.end_time()]
    metrics = calc_metrics(actual, last_forecast) if add_metrics else None

    return LastPrediction(actual=actual, prediction=last_forecast, lookback=lookback, metrics=metrics)
