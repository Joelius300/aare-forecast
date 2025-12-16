from datetime import tzinfo
import json
import logging
import pickle
from concurrent.futures.process import ProcessPoolExecutor
from typing import Literal, Mapping, Optional, Sequence, cast, overload

import pandas as pd

from aare.constants import TEMP
import numpy as np
import torch
from darts import TimeSeries
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
from aare.utils import FORECAST_SAMPLES_FOLDER, METRICS_FOLDER, get_context_len, relocalize_times

logger = logging.getLogger(__name__)

metrics_idx = {"MAE": 0, "RMSE": 1, "MADPD": 2}
metric_names = list(metrics_idx.keys())
MetricType = Literal["MAE", "RMSE", "MADPD"]


@overload
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state=42,
    get_raw: Literal[False] = False,
    run_ts_delta=pd.Timedelta(1, "s"),
) -> tuple[EvalMetric, ForecastSamples]:
    pass


@overload
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state=42,
    get_raw: Literal[True],
    run_ts_delta=pd.Timedelta(1, "s"),
) -> tuple[EvalMetric, ForecastSamples, pd.DataFrame]:
    pass


# TODO Unit test
# TODO refactor completely; evaluation should happen on historical forecasts in the same table format as they are
#  stored during inference. This way you only need to call historical_forecasts and transform them into the table format
#  to use all the fancy evaluation and potentially reporting functionality designed for both past and continuous validation.
#  The forecasts dataframe can be joined with ground truth for comparison (evaluation). The evaluation logic should be
#  pandas or polars, not using darts metrics and numpy. This will be much faster and more agnostic=useful. Use TDD for this.
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride=24,
    min_lookback_hours=-1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state=42,
    get_raw: bool = False,
    run_ts_delta=pd.Timedelta(1, "s"),
) -> tuple[EvalMetric, ForecastSamples] | tuple[EvalMetric, ForecastSamples, pd.DataFrame]:
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
    Optionally get the raw err and dpd metrics.
    """
    if isinstance(val, TimeSeries):
        if future_cov is not None:
            logger.warning("Passed future_cov but not val_subs, so splits will most likely be incompatible!")

        val_subs = extract_subseries(val, mode="any")
    else:
        assert isinstance(val, list), "val must be a list or a TimeSeries"
        val_subs = val

    (
        (
            metrics,
            samples,
        ),
        raw_metrics,
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
        run_ts_delta=run_ts_delta,
    )

    lookback_hours = max(get_context_len(model), min_lookback_hours)
    sample = ForecastSamples(
        *(
            EvalForecast(val, fc, lookback_hours, EvalMetric.from_ndarray(fc_m), future_cov=future_cov)
            for fc, fc_m in samples
        )
    )

    if get_raw:
        assert raw_metrics is not None, "raw metrics were none?!"
        return metrics, sample, raw_metrics

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
    tz: str | tzinfo,
    run_ts_delta: pd.Timedelta,
):
    hf = historical_forecasts(
        model, val, horizon, stride, parallel, verbose, future_cov, num_samples, data_transformers, random_state
    )
    hf = [fc for fc_l in hf for fc in fc_l]

    hf_df = hf_to_table(hf, tz, run_ts_delta)
    hf_df = join_true_data(hf_df, val, tz)
    hf_df["err"] = get_err(hf_df)
    hf_df["dpd"] = get_dpd(hf_df)
    metric_df = get_metrics(hf_df)

    return get_samples(hf, metric_df, metric), hf_df


def get_samples(hf: list[TimeSeries], metric_df: pd.DataFrame, metric: MetricType = "MAE"):
    """Get best, worst and most avg samples together with aggregated median metrics."""
    assert len(hf) == metric_df.shape[0], "historical forecast don't match metric df"

    metric_df = metric_df[metric_names]
    metrics_median = np.median(metric_df, axis=0).astype(float)
    metrics_std = np.std(metric_df, axis=0, dtype=float)
    agg_metrics = EvalMetric.from_ndarray(metrics_median, metrics_std)

    worst_index, best_index = np.argmax(metric_df, axis=0), np.argmin(metric_df, axis=0)
    most_avg_index = np.argmin(np.abs(metric_df - metrics_median), axis=0)
    assert (
        worst_index.shape == (len(metric_names),)
        and best_index.shape == (len(metric_names),)
        and most_avg_index.shape == (len(metric_names),)
    ), "worst, best, most_avg index reduction is faulty"

    metric_i = metrics_idx[metric]
    worst_index, best_index, most_avg_index = worst_index[metric_i], best_index[metric_i], most_avg_index[metric_i]

    return (
        agg_metrics,
        (
            (hf[most_avg_index], metric_df.iloc[most_avg_index]),
            (hf[best_index], metric_df.iloc[best_index]),
            (hf[worst_index], metric_df.iloc[worst_index]),
        ),
    )


def hf_to_table(hf: list[TimeSeries], tz: str | tzinfo, run_ts_delta=pd.Timedelta(1, "s")) -> pd.DataFrame:
    """Convert a list of historical forecasts to table format with run_ts, time and pred. tz needed to make times aware."""
    times = np.stack([fc.time_index.values for fc in hf])

    run_ts = times[:, 0] - run_ts_delta
    run_ts = run_ts.repeat(times.shape[1])

    pred = np.concat([fc.values() for fc in hf])

    hf_df = pd.DataFrame(pred, index=times.ravel(), columns=["pred"])
    hf_df = hf_df.reset_index(names="time")
    hf_df.insert(0, "run_ts", run_ts)  # add run_ts as first col

    hf_df["time"] = relocalize_times(hf_df["time"], tz)
    hf_df["run_ts"] = relocalize_times(hf_df["run_ts"], tz)

    return hf_df


def join_true_data(hf_df: pd.DataFrame, true_target: list[TimeSeries] | pd.DataFrame, tz: str | tzinfo):
    """
    Add 'actual' column to the historical forecast dataframe for evaluation. Either supply a dataframe directly
    or a list of time series with the target value (e.g. val slices). tz needed to make true times aware.
    """
    if isinstance(true_target, list):
        true_target = pd.concat([v.to_dataframe() for v in true_target])

    if TEMP in true_target.columns:
        true_target = true_target.rename(columns={TEMP: "actual"})

    assert "actual" in true_target.columns, "'actual' column missing in true df"

    if "time" not in true_target.columns:
        true_target = true_target.reset_index(names="time")

    # make sure joining with aware timestamps
    true_target["time"] = relocalize_times(true_target["time"], tz)

    return hf_df.join(true_target[["time", "actual"]].set_index("time"), on="time")


def get_dpd(hf_df: pd.DataFrame):
    """Calculate the raw dpd metric from the 'actual' and 'pred' columns together with 'time'."""
    assert hf_df["time"].dt.tz is not None, "Naive timestamps in dpd, will likely introduce alignment errors!"
    # group by run and date of the predicted time
    daily_max = hf_df[["actual", "pred"]].groupby([hf_df["run_ts"], hf_df["time"].dt.date]).transform("max")

    return daily_max["actual"] - daily_max["pred"]


def get_err(hf_df: pd.DataFrame):
    """Calculate the raw y_true - y_pred errors."""
    return hf_df["actual"] - hf_df["pred"]


def get_metrics(hf_df: pd.DataFrame):
    """Calculate MAE, RMSE and MADPD per forecast."""
    x = hf_df[["run_ts", "err", "dpd"]].copy()
    x["err_sq"] = x["err"] ** 2
    x[["err", "dpd"]] = x[["err", "dpd"]].abs()

    df_agg = x.groupby("run_ts").agg("mean").rename(columns=dict(err="MAE", err_sq="RMSE", dpd="MADPD"))
    df_agg["RMSE"] = np.sqrt(df_agg["RMSE"])

    return df_agg


# TODO extract into own file for historical forecasts
def historical_forecasts(
    model: ForecastingModel,
    val: list[TimeSeries],
    horizon: int,
    stride: int,
    parallel: bool | int | Literal["auto"] = False,
    verbose=False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples=128,
    data_transformers: Optional[DataTransformers] = None,
    random_state: Optional[int] = 42,
) -> list[list[TimeSeries]]:
    """Simulate historical forecasts with a pre-trained model on one or more validation series, optionally in parallel."""
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
    return _historical_forecasts_parallel(
        model, val, stride, horizon, parallel, verbose, future_cov, num_samples, data_transformers, random_state
    )


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
        # TODO could add a guard that it still uses parallelization if horizon > output_chunk_length
        #  because AR is never optimized IIRC.
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
    tz: str | tzinfo,
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
