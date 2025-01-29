import json
import pickle
from typing import cast, Mapping

import numpy as np
import torch
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
from darts.models.forecasting.global_baseline_models import _GlobalNaiveModel
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.evaluation.forecast import Forecast
from aare.evaluation.metrics import Metrics
from aare.params import ValidationParams
from aare.preparation import prepare_ts
from aare.utils import METRICS_FOLDER, LAST_FORECASTS_FOLDER, get_context_len


def evaluate_model(
    model: GlobalForecastingModel, val: list[TimeSeries], stride: int, horizon: int
) -> tuple[Metrics, TimeSeries]:
    """
    Evaluates a global model on multiple validation series with a specified stride and forecast horizon.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc.

    Returns the aggregated metrics and the last prediction the model made.
    """
    if isinstance(model, _GlobalNaiveModel):
        # only takes the components etc. global naive don't care about the values
        model.fit(val[0])

    # produces multiple predictions (TimeSeries) with the specified stride FOR EACH SUBSERIES
    historical_forecasts = cast(
        list[list[TimeSeries]],
        # (most likely) uses extreme_lags to find where to start forecasting
        model.historical_forecasts(
            val,
            stride=stride,
            forecast_horizon=horizon,
            last_points_only=False,
            retrain=False,
        ),
    )

    backtest = model.backtest(val, historical_forecasts=historical_forecasts, metric=[mae, rmse])

    metrics = np.mean(backtest, axis=0, dtype=float)
    metrics = Metrics(mae=metrics[0], rmse=metrics[1])

    # TODO also return best and worst prediction (will need reduction=None in backtest)
    last_prediction = historical_forecasts[-1][-1]

    return metrics, last_prediction


def evaluation_pipeline(
    models: Mapping[str, GlobalForecastingModel], forecast_horizon: int, validation_params: ValidationParams
) -> None:
    """Evaluate all specified models on the validation data and write the results to the pre-defined folders."""
    dataset = AareDataset.from_conf()
    stride = validation_params["stride"]
    min_lookback_hours = validation_params["min_lookback_hours"]
    val = prepare_ts(dataset.get_val())
    val_subs = extract_subseries(val)

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")

    METRICS_FOLDER.mkdir(exist_ok=True)
    LAST_FORECASTS_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, last_prediction = evaluate_model(model, val_subs, stride, forecast_horizon)
        lookback_hours = max(get_context_len(model), min_lookback_hours)
        last_forecast = Forecast(val, last_prediction, lookback_hours)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(LAST_FORECASTS_FOLDER / f"{name}.pkl", "wb") as last_forecast_file:
            pickle.dump(last_forecast, last_forecast_file)
