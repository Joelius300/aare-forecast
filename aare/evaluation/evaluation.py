from typing import cast

import numpy as np
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.global_baseline_models import _GlobalNaiveModel

from aare.evaluation.metrics import Metrics


def evaluate_model(
    model: GlobalForecastingModel, val: list[TimeSeries], stride: int, horizon: int
) -> tuple[Metrics, TimeSeries]:
    """
    Evaluates a global model on multiple validation series with a specified stride and forecast horizon.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc.

    Returns the aggregated metrics and the last forecast the model made.
    """
    if isinstance(model, _GlobalNaiveModel):
        # only takes the components etc. global naive don't care about the values
        model.fit(val[0])

    # produces multiple predictions (TimeSeries) with the specified stride FOR EACH SUBSERIES
    historical_forecasts = cast(
        list[list[TimeSeries]],
        model.historical_forecasts(
            val,
            stride=stride,
            forecast_horizon=horizon,
            last_points_only=False,
            retrain=False,
        ),
    )

    backtest = model.backtest(val, historical_forecasts=historical_forecasts, metric=[mae, rmse])

    metrics = np.mean(backtest, axis=0)
    metrics = Metrics(mae=float(metrics[0]), rmse=float(metrics[1]))

    last_forecast = historical_forecasts[-1][-1]

    return metrics, last_forecast
