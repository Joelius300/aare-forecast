import json
import pickle
from typing import cast, Mapping, Literal, Optional

import numpy as np
import torch
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.global_baseline_models import _GlobalNaiveModel
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.evaluation.forecast import Forecast
from aare.evaluation.forecast_samples import ForecastSamples
from aare.evaluation.metrics import Metrics
from aare.params import ValidationParams
from aare.preparation import prepare_ts
from aare.utils import METRICS_FOLDER, FORECAST_SAMPLES_FOLDER, get_context_len


def _evaluate_model(
    model: ForecastingModel, val: list[TimeSeries], horizon: int, stride: int, *, metric: Literal["MAE", "RMSE"] = "MAE"
) -> tuple[Metrics, tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray]]:
    """
    Evaluates a forecasting model on multiple validation series with a specified stride and forecast horizon.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc.

    Returns the aggregated metrics as well as the last, best and worst prediction the model made (decided by MAE).
    """
    # TODO implement parallelization. Esp. for CPU bound models, you could easily spin up multiple processes to
    # to speed up the predictions. Maybe there's even a smart scheduling option to use the length of the subseries
    # as weight basically (you would want the longest running ones to start first).
    # Also, don't parallelize if the model supports_optimized_historical_forecasts or whatever,
    # then it would waste time probably? at least warn the user that the config is prob bad.
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
    metric_i = 0 if metric == "MAE" else (1 if metric == "RMSE" else None)
    if metric_i is None:
        raise ValueError(f"Invalid metric '{metric}'")

    # flatten so all historical forecasts are treated the same, despite multiple val series.
    # this is really important because otherwise, short series with only a few evaluations
    # are weighed the same as very long series with much more samples -> not all samples
    # are weighted equally overall.
    backtest_flat = np.concatenate(backtest, axis=0)
    historical_forecasts_flat = [ts for ll in historical_forecasts for ts in ll]

    worst_index, best_index = np.argmax(backtest_flat, axis=0), np.argmin(backtest_flat, axis=0)
    assert worst_index.shape == (2,) and best_index.shape == (2,), "worst, best index reduction is faulty"
    worst_index, best_index = worst_index[metric_i], best_index[metric_i]
    # TODO could check and warn if MAE and RMSE result in different best/worst

    metrics_median = np.median(backtest_flat, axis=0).astype(float)
    metrics_std = np.std(backtest_flat, axis=0, dtype=float)
    assert metrics_median.shape == (2,) and metrics_std.shape == (2,), "metric reduction is faulty"
    metrics = Metrics(mae=metrics_median[0], rmse=metrics_median[1], mae_std=metrics_std[0], rmse_std=metrics_std[1])

    return (
        metrics,
        (historical_forecasts_flat[-1], backtest_flat[-1]),
        (historical_forecasts_flat[best_index], backtest_flat[best_index]),
        (historical_forecasts_flat[worst_index], backtest_flat[worst_index]),
    )


def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries,
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    metric: Literal["MAE", "RMSE"] = "MAE",
    val_subs: Optional[list[TimeSeries]] = None,
):
    """
    Evaluates a forecasting model on a validation series with a specified stride and forecast horizon.
    The series will be split on all gaps where a NaN is present in any component using extract_subseries.
    If you already have it ready or want to avoid that, pass val_subs.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc. all other models are
    expected to be trained/fitted already.

    Returns the aggregated metrics as well as the last, best and worst prediction
    the model made (decided by MAE or whatever you specify).
    """
    if not val_subs:
        val_subs = extract_subseries(val, mode="any")

    (
        metrics,
        (last_prediction, last_prediction_m),
        (best_prediction, best_prediction_m),
        (worst_prediction, worst_prediction_m),
    ) = _evaluate_model(model, val_subs, horizon, stride, metric=metric)
    lookback_hours = max(get_context_len(model), min_lookback_hours)

    last_forecast = Forecast(val, last_prediction, lookback_hours, Metrics.from_ndarray(last_prediction_m))
    best_forecast = Forecast(val, best_prediction, lookback_hours, Metrics.from_ndarray(best_prediction_m))
    worst_forecast = Forecast(val, worst_prediction, lookback_hours, Metrics.from_ndarray(worst_prediction_m))
    sample = ForecastSamples(last_forecast, best_forecast, worst_forecast)

    return metrics, sample


def evaluation_pipeline(
    models: Mapping[str, ForecastingModel], forecast_horizon: int, validation_params: ValidationParams
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
        metrics, sample = evaluate_model(model, val, forecast_horizon, stride, min_lookback_hours, val_subs=val_subs)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(FORECAST_SAMPLES_FOLDER / f"{name}.pkl", "wb") as forecast_sample_file:
            pickle.dump(sample, forecast_sample_file)
