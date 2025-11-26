from datetime import tzinfo
import json
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
from typing import Literal, Mapping, Optional, Sequence, cast

from aare.evaluation.custom_metrics.daily_peak import madpd
import numpy as np
import torch
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.global_baseline_models import _GlobalNaiveModel
from darts.utils.missing_values import extract_subseries

from aare.AareDataset import AareDataset
from aare.compat.types import DataTransformers
from aare.evaluation.eval_forecast import EvalForecast
from aare.evaluation.forecast_samples import ForecastSamples
from aare.evaluation.eval_metric import EvalMetric
from aare.params import ValidationParams
from aare.preparation import prepare_ts_aare_temp
from aare.utils import FORECAST_SAMPLES_FOLDER, METRICS_FOLDER, get_context_len

logger = logging.getLogger(__name__)

used_metrics = [mae, rmse, madpd]
metrics_idx = {"MAE": 0, "RMSE": 1, "MADPD": 2}
MetricType = Literal["MAE", "RMSE", "MADPD"]


def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    metric: Literal["MAE", "RMSE"] = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state=42,
    tz: str | tzinfo | None = None,
):
    """
    Evaluates a forecasting model on a validation series with a specified stride and forecast horizon.
    Pass a list of subseries without NaNs. If passing a single series, it will be split using extract_subseries.

    The covariates don't need to be sliced exactly, it will use the overlap of the val_subs with the covariates.
    If the future_cov has gaps, make sure that you pass val_subs with only subs where the future_cov has complete data.

    Global Naive Models are "trained" first to give them knowledge about the dimensions etc. all other models are
    expected to be trained/fitted already.

    Allows for parallelization, but beware that it will replicate the model on multiple processes, so it has to be
    pickleable, and it will multiply memory usage.

    Random state is fixed at 42 by default for reproducibility.

    The timezone (tz) is used to calculate the mean absolute daily peak difference metric.

    Returns the aggregated metrics as well as the most average, best and worst forecast
    the model made (decided by MAE or whatever you specify).
    """
    if isinstance(val, TimeSeries):
        if future_cov is not None:
            logger.warning("Passed future_cov but not val_subs, so splits will most likely be incompatible!")

        val_subs = extract_subseries(val, mode="any")
    else:
        assert isinstance(val, list), "val must be a list or a TimeSeries"
        val_subs = val

    (
        metrics,
        (most_avg_forecast, most_avg_forecast_m),
        (best_forecast, best_forecast_m),
        (worst_forecast, worst_forecast_m),
    ) = _evaluate_model(
        model,
        val_subs,
        horizon,
        stride,
        metric=metric,
        parallel=parallel,
        verbose=verbose,
        future_cov=future_cov,
        num_samples=num_samples,
        data_transformers=data_transformers,
        random_state=random_state,
        tz=tz,
    )
    lookback_hours = max(get_context_len(model), min_lookback_hours)

    most_avg_forecast = EvalForecast(
        val, most_avg_forecast, lookback_hours, EvalMetric.from_ndarray(most_avg_forecast_m), future_cov=future_cov
    )
    best_forecast = EvalForecast(
        val, best_forecast, lookback_hours, EvalMetric.from_ndarray(best_forecast_m), future_cov=future_cov
    )
    worst_forecast = EvalForecast(
        val, worst_forecast, lookback_hours, EvalMetric.from_ndarray(worst_forecast_m), future_cov=future_cov
    )

    sample = ForecastSamples(most_avg_forecast, best_forecast, worst_forecast)

    return metrics, sample


def _evaluate_model(
    model: ForecastingModel,
    val: list[TimeSeries],
    horizon: int,
    stride: int,
    metric: MetricType,
    parallel: bool | int | Literal["auto"],
    verbose: bool,
    future_cov: TimeSeries | Sequence[TimeSeries] | None,
    num_samples: int,
    data_transformers: Optional[DataTransformers],
    random_state: Optional[int],
    tz: str | tzinfo | None,
) -> tuple[EvalMetric, tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray]]:
    historical_forecasts = _historical_forecasts(
        model, val, stride, horizon, parallel, verbose, future_cov, num_samples, data_transformers, random_state
    )

    metric_i = metrics_idx.get(metric)
    if metric_i is None:
        raise ValueError(f"Invalid metric '{metric}'")

    return _get_agg_metrics(model, historical_forecasts, val, metric_i, tz)


def _get_agg_metrics(
    model: ForecastingModel,
    historical_forecasts: list[list[TimeSeries]],
    val: list[TimeSeries],
    metric_i: int,
    tz: str | tzinfo | None,
):
    # run metric calculations on all historical forecasts
    backtest = cast(
        list[np.ndarray],
        model.backtest(
            val,
            historical_forecasts=historical_forecasts,
            metric=used_metrics,
            # must align order of kwargs with order of metrics
            metric_kwargs=[{}, {}, dict(tz=tz)],
            reduction=None,
        ),
    )

    # flatten so all historical forecasts are treated the same, despite multiple val series.
    # this is really important because otherwise, short series with only a few evaluations
    # are weighed the same as very long series with much more samples -> not all samples
    # are weighted equally overall.
    backtest_flat = cast(np.ndarray, np.concatenate(backtest, axis=0))
    historical_forecasts_flat = [ts for ll in historical_forecasts for ts in ll]

    # the function choice here (median, not mean) is written as info in the Metrics class.
    metrics_median = np.median(backtest_flat, axis=0).astype(float)
    metrics_std = np.std(backtest_flat, axis=0, dtype=float)
    assert metrics_median.shape == (len(used_metrics),) and metrics_std.shape == (len(used_metrics),), (
        "metric reduction is faulty"
    )
    agg_metrics = EvalMetric.from_ndarray(metrics_median, metrics_std)

    worst_index, best_index = np.argmax(backtest_flat, axis=0), np.argmin(backtest_flat, axis=0)
    most_avg_index = np.argmin(np.abs(backtest_flat - metrics_median), axis=0)
    assert (
        worst_index.shape == (len(used_metrics),)
        and best_index.shape == (len(used_metrics),)
        and most_avg_index.shape == (len(used_metrics),)
    ), "worst, best, most_avg index reduction is faulty"
    worst_index, best_index, most_avg_index = worst_index[metric_i], best_index[metric_i], most_avg_index[metric_i]
    # TODO could check and warn if MAE and RMSE result in different best/worst

    return (
        agg_metrics,
        (historical_forecasts_flat[most_avg_index], backtest_flat[most_avg_index]),
        (historical_forecasts_flat[best_index], backtest_flat[best_index]),
        (historical_forecasts_flat[worst_index], backtest_flat[worst_index]),
    )


def _historical_forecasts(
    model: ForecastingModel,
    val: list[TimeSeries],
    stride: int,
    horizon: int,
    parallel: bool | int | Literal["auto"],
    verbose: bool,
    future_cov: TimeSeries | Sequence[TimeSeries] | None,
    num_samples: int,
    data_transformers: DataTransformers | None,
    random_state: int | None,
) -> list[list[TimeSeries]]:
    # does some validation and prep, then uses _historical_forecasts_parallel
    if len(val) == 0:
        raise ValueError("Must pass at least one validation series")

    if not model.supports_transferable_series_prediction:
        raise ValueError("Cannot evaluate a model which doesn't support transferable prediction.")

    # could break in any release since it's not public api
    if not model._supports_non_retrainable_historical_forecasts:
        # currently (24.03) the only models that support transferable series prediction but not
        # non-retrainable historical forecasts are local ensemble models IIRC.
        raise ValueError("Cannot evaluate a model which doesn't support non-retrainable historical forecasts.")

    parallel = _validate_parallel(parallel, model, val)

    if isinstance(model, _GlobalNaiveModel):
        # only takes the components etc. global naive don't care about the values
        model.fit(val[0])

    # simulate historical forecasts (without retraining!)
    historical_forecasts = _historical_forecasts_parallel(
        model, val, stride, horizon, parallel, verbose, future_cov, num_samples, data_transformers, random_state
    )

    return historical_forecasts


def _historical_forecasts_parallel(
    model: ForecastingModel,
    val: list[TimeSeries],
    stride: int,
    horizon: int,
    parallel: bool | int,
    verbose: bool,
    future_cov: TimeSeries | Sequence[TimeSeries] | None,
    num_samples: int,
    data_transformers: Optional[DataTransformers],
    random_state: Optional[int],
) -> list[list[TimeSeries]]:
    if (
        future_cov is not None
        and not isinstance(future_cov, TimeSeries)
        and isinstance(future_cov, Sequence)
        and len(future_cov) != len(val)
    ):
        raise ValueError("When passing future_cov as a list, it must have the same number of entries as val")

    actual_num_samples = num_samples if model.supports_probabilistic_prediction else 1

    if not parallel:
        return cast(
            list[list[TimeSeries]],
            # (most likely) uses extreme_lags to find where to start forecasting
            # produces multiple forecasts (TimeSeries) with the specified stride FOR EACH SUBSERIES
            model.historical_forecasts(
                val,  # passing multiple ts so we get multiple sets of forecasts back
                # future_covariates can handle slices like val but also just a big chunk with the relevant data
                future_covariates=future_cov,
                stride=stride,
                forecast_horizon=horizon,
                last_points_only=False,
                retrain=False,
                num_samples=actual_num_samples,
                verbose=verbose,  # seemingly only for retraining, so probably useless
                data_transformers=data_transformers,
                random_state=random_state,
            ),
        )

    # keep track of the index so order can be reconstructed
    val_idx = list(enumerate(val))
    # longest series first
    prioritized = sorted(val_idx, key=lambda i_ts: len(i_ts[1]), reverse=True)

    # prepare covariates for both split and unified series
    if future_cov is None or isinstance(future_cov, TimeSeries):
        future_covs = [future_cov] * len(prioritized)
    else:
        assert isinstance(future_cov, Sequence), "future_cov is not a sequence?!"
        future_covs = []
        for i, _ in prioritized:
            future_covs.append(future_cov[i])

    logger.debug(f"Creating historical forecasts for [{', '.join((str(len(i_ts[1])) for i_ts in prioritized))}]")

    with ProcessPoolExecutor(max_workers=None if parallel is True else parallel) as pool:
        hf = pool.map(
            _parallel_forecast_step,
            prioritized,
            [model] * len(prioritized),
            [stride] * len(prioritized),
            [horizon] * len(prioritized),
            [actual_num_samples] * len(prioritized),
            future_covs,
            [data_transformers] * len(prioritized),
            [random_state] * len(prioritized),
        )

        hf = list(hf)  # makes it easier and the overhead is nothing
        # restore original order
        return [i_ts[1] for i_ts in sorted(hf, key=lambda iv: iv[0])]


def _parallel_forecast_step(
    i_ts: tuple[int, TimeSeries],
    model: ForecastingModel,
    stride: int,
    horizon: int,
    num_samples: int,
    future_cov: TimeSeries | None,
    data_transformers: Optional[DataTransformers],
    random_state: Optional[int],
):
    assert future_cov is None or isinstance(future_cov, TimeSeries), "Invalid type of future_cov"
    i, ts = i_ts
    forecasts = model.historical_forecasts(
        ts,  # passing now a single ts -> get a single set of forecasts
        future_covariates=future_cov,  # must now be a single series
        stride=stride,
        forecast_horizon=horizon,
        last_points_only=False,
        retrain=False,
        num_samples=num_samples,
        verbose=False,  # no need in another process
        data_transformers=data_transformers,
        random_state=random_state,
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


def evaluation_pipeline_uni(
    models: Mapping[str, ForecastingModel],
    forecast_horizon: int,
    validation_params: ValidationParams,
    tz: str | tzinfo | None,
) -> None:
    """Evaluate all specified models on the validation data and write the results to the pre-defined folders."""
    dataset = AareDataset.from_conf()
    stride = validation_params["stride"]
    min_lookback_hours = validation_params["min_lookback_hours"]
    val = prepare_ts_aare_temp(dataset.get_val())
    val_subs = extract_subseries(val)

    # mostly to suppress the torch notice, darts has bad support for this
    torch.set_float32_matmul_precision("medium")

    METRICS_FOLDER.mkdir(exist_ok=True)
    FORECAST_SAMPLES_FOLDER.mkdir(exist_ok=True)

    for name, model in models.items():
        metrics, sample = evaluate_model(model, val_subs, forecast_horizon, stride, min_lookback_hours, tz=tz)

        with open(METRICS_FOLDER / f"{name}.json", "wt") as metrics_file:
            json.dump(metrics.to_dict(), metrics_file)

        with open(FORECAST_SAMPLES_FOLDER / f"{name}.pkl", "wb") as forecast_sample_file:
            pickle.dump(sample, forecast_sample_file)
