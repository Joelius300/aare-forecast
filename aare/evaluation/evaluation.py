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
from aare.evaluation.forecast_samples import ForecastSamples
from aare.evaluation.metrics import Metrics
from aare.params import ValidationParams
from aare.preparation import prepare_ts
from aare.utils import METRICS_FOLDER, FORECAST_SAMPLES_FOLDER, get_context_len


def evaluate_model(
    model: GlobalForecastingModel, val: list[TimeSeries], stride: int, horizon: int
) -> tuple[Metrics, tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray]]:
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

    backtest = cast(
        list[np.ndarray],
        model.backtest(val, historical_forecasts=historical_forecasts, metric=[mae, rmse], reduction=None),
    )

    # flatten so all historical forecasts are treated the same, despite multiple val series.
    # this is really important because otherwise, short series with only a few evaluations
    # are weighed the same as very long series with much more samples -> not all samples
    # are weighted equally overall.
    backtest_flat = np.concatenate(backtest, axis=0)
    historical_forecasts_flat = [ts for ll in historical_forecasts for ts in ll]

    worst_index, best_index = np.argmax(backtest_flat, axis=0), np.argmin(backtest_flat, axis=0)
    assert worst_index.shape == (2,) and best_index.shape == (2,), "worst, best index reduction is faulty"
    worst_index, best_index = worst_index[0], best_index[0]  # MAE as decisive metric
    # TODO could check and warn if MAE and RMSE have different best/worst

    metrics_mean = np.mean(backtest_flat, axis=0, dtype=float)
    metrics_std = np.std(backtest_flat, axis=0, dtype=float)
    assert metrics_mean.shape == (2,) and metrics_std.shape == (2,), "metric reduction is faulty"
    metrics = Metrics(mae=metrics_mean[0], rmse=metrics_mean[1], mae_std=metrics_std[0], rmse_std=metrics_std[1])

    return (
        metrics,
        (historical_forecasts_flat[-1], backtest_flat[-1]),
        (historical_forecasts_flat[best_index], backtest_flat[best_index]),
        (historical_forecasts_flat[worst_index], backtest_flat[worst_index]),
    )


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
    FORECAST_SAMPLES_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        (
            metrics,
            (last_prediction, last_prediction_m),
            (best_prediction, best_prediction_m),
            (worst_prediction, worst_prediction_m),
        ) = evaluate_model(model, val_subs, stride, forecast_horizon)
        lookback_hours = max(get_context_len(model), min_lookback_hours)

        last_forecast = Forecast(val, last_prediction, lookback_hours, Metrics.from_ndarray(last_prediction_m))
        best_forecast = Forecast(val, best_prediction, lookback_hours, Metrics.from_ndarray(best_prediction_m))
        worst_forecast = Forecast(val, worst_prediction, lookback_hours, Metrics.from_ndarray(worst_prediction_m))
        sample = ForecastSamples(last_forecast, best_forecast, worst_forecast)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(FORECAST_SAMPLES_FOLDER / f"{name}.pkl", "wb") as forecast_sample_file:
            pickle.dump(sample, forecast_sample_file)
