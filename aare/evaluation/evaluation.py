import json
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
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

logger = logging.getLogger(__name__)


def _evaluate_model(
    model: ForecastingModel,
    val: list[TimeSeries],
    horizon: int,
    stride: int,
    metric: Literal["MAE", "RMSE"] = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
) -> tuple[Metrics, tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray]]:
    if len(val) == 0:
        raise ValueError("Must pass at least one validation series")

    if not model.supports_transferrable_series_prediction:
        raise ValueError("Cannot evaluate a model which doesn't support transferrable prediction.")

    # could break in any release since it's not public api
    if not model._supports_non_retrainable_historical_forecasts:
        # currently (24.03) the only models that support transferrable series prediction but not
        # non-retrainable historical forecasts are local ensemble models IIRC.
        raise ValueError("Cannot evaluate a model which doesn't support non-retrainable historical forecasts.")

    parallel = _validate_parallel(parallel, model, val)

    if isinstance(model, _GlobalNaiveModel):
        # only takes the components etc. global naive don't care about the values
        model.fit(val[0])

    # simulate historical forecasts (without retraining!)
    historical_forecasts = _historical_forecasts_parallel(model, val, stride, horizon, parallel, verbose)

    # run metric calculations on all those forecasts
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


def _historical_forecasts_parallel(
    model: ForecastingModel, val: list[TimeSeries], stride: int, horizon: int, parallel: bool | int, verbose: bool
) -> list[list[TimeSeries]]:
    if not parallel:
        return cast(
            list[list[TimeSeries]],
            # (most likely) uses extreme_lags to find where to start forecasting
            # produces multiple predictions (TimeSeries) with the specified stride FOR EACH SUBSERIES
            model.historical_forecasts(
                val,  # passing multiple ts so we get multiple sets of forecasts back
                stride=stride,
                forecast_horizon=horizon,
                last_points_only=False,
                retrain=False,
            ),
        )

    # keep track of the index so order can be reconstructed
    val_idx = list(enumerate(val))
    # longest series first
    prioritized = sorted(val_idx, key=lambda i_ts: len(i_ts[1]), reverse=True)
    logger.debug(f"Creating historical forecasts for [{', '.join((str(len(i_ts[1])) for i_ts in prioritized))}]")

    with ProcessPoolExecutor(max_workers=None if parallel is True else parallel) as pool:
        hf = pool.map(
            _parallel_forecast_step,
            prioritized,
            [model] * len(prioritized),
            [stride] * len(prioritized),
            [horizon] * len(prioritized),
        )

        hf = list(hf)  # makes it easier and the overhead is nothing
        # restore original order
        return [i_ts[1] for i_ts in sorted(hf, key=lambda iv: iv[0])]


def _parallel_forecast_step(i_ts: tuple[int, TimeSeries], model: ForecastingModel, stride: int, horizon: int):
    i, ts = i_ts
    forecasts = model.historical_forecasts(
        ts,  # passing now a single ts -> get a single set of forecasts
        stride=stride,
        forecast_horizon=horizon,
        last_points_only=False,
        retrain=False,
        verbose=False,  # no need in another process
    )

    return i, cast(list[TimeSeries], forecasts)


def _validate_parallel(parallel: bool | int | Literal["auto"], model: ForecastingModel, val: list) -> bool | int:
    # Thank you, Python, you did it again... bool is a subclass of int OMFG
    if parallel == "auto":
        # not sure if this is a good heuristic, but probably not too bad for us
        parallel = False if model.supports_optimized_historical_forecasts else True
    elif parallel is False:
        if not model.supports_optimized_historical_forecasts and len(val) > 1:
            logger.warning(
                "Model does not support optimized historical forecasts and you're evaluating multiple "
                "validation series; you might benefit from parallelized evaluation."
            )
    elif parallel is True or type(parallel) is int:
        if model.supports_optimized_historical_forecasts:
            logger.warning(
                "Model already supports optimized forecasts, unclear if parallelization improves performance"
            )
        if len(val) == 1:
            logger.warning("Only evaluating on a single validation slice, parallelization does not make sense.")
            parallel = False
        if type(parallel) is int and parallel < 2:
            raise ValueError("Parallelization must be of degree 2 or higher")
    else:
        raise ValueError(f"Invalid option for parallel: {parallel}")

    # now parallel can only be True, False or an int > 2
    assert type(parallel) in (bool, int), "parallel is something other than bool or int?!"

    return cast(bool | int, parallel)


def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries,
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    val_subs: Optional[list[TimeSeries]] = None,
    metric: Literal["MAE", "RMSE"] = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
):
    """
    Evaluates a forecasting model on a validation series with a specified stride and forecast horizon.
    The series will be split on all gaps where a NaN is present in any component using extract_subseries.
    If you already have it ready or want to avoid that, pass val_subs.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc. all other models are
    expected to be trained/fitted already.

    Allows for parallelization, but beware that it will replicate the model on multiple processes, so it has to be
    pickleable, and it will multiply memory usage.

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
    ) = _evaluate_model(model, val_subs, horizon, stride, metric=metric, parallel=parallel, verbose=verbose)
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
