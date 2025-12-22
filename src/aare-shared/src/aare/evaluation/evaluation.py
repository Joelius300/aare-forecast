from datetime import tzinfo
import logging
from typing import Literal, overload, cast
from collections.abc import Sequence

import pandas as pd

import numpy as np
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.utils.missing_values import extract_subseries

from aare.compat.types import DataTransformers
from aare.evaluation.eval_forecast import EvalForecast
from aare.evaluation.forecast_samples import ForecastSamples
from aare.evaluation.eval_metric import EvalMetric
from aare.evaluation.historical_forecasts import historical_forecasts
from aare.utils import relocalize_times
from aare.darts_utils import get_context_len

logger = logging.getLogger(__name__)

metrics_idx = {"MAE": 0, "RMSE": 1, "MADPD": 2}
metric_names = list(metrics_idx.keys())
MetricType = Literal["MAE", "RMSE", "MADPD"]


@overload
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride: int = 24,
    min_lookback_hours: int = -1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose: bool = False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples: int = 128,
    data_transformers: DataTransformers | None = None,
    random_state: int = 42,
    get_raw: Literal[False] = False,
    run_ts_delta: pd.Timedelta = pd.Timedelta(1, "s"),
) -> tuple[EvalMetric, ForecastSamples]:
    pass


@overload
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride: int = 24,
    min_lookback_hours: int = -1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose: bool = False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples: int = 128,
    data_transformers: DataTransformers | None = None,
    random_state: int = 42,
    get_raw: Literal[True],
    run_ts_delta: pd.Timedelta = pd.Timedelta(1, "s"),
) -> tuple[EvalMetric, ForecastSamples, pd.DataFrame]:
    pass


# TODO Unit test
def evaluate_model(
    model: ForecastingModel,
    val: TimeSeries | list[TimeSeries],
    horizon: int,
    stride: int = 24,
    min_lookback_hours: int = -1,
    *,
    tz: str | tzinfo,
    metric: MetricType = "MAE",
    parallel: bool | int | Literal["auto"] = False,
    verbose: bool = False,
    future_cov: TimeSeries | Sequence[TimeSeries] | None = None,
    num_samples: int = 128,
    data_transformers: DataTransformers | None = None,
    random_state: int = 42,
    get_raw: bool = False,
    run_ts_delta: pd.Timedelta = pd.Timedelta(1, "s"),
    month_filter: tuple[int, int] | None = None,
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

    The timezone (tz) is used to relocalize the times and calculate the mean absolute daily peak difference metric.

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

    # TODO add another argument to remove out of season periods in val before evaluating it -> this also automatically
    #  improves parallelization without having to make sure the overlaps are perfect.

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
        month_filter=month_filter,
    )

    lookback_hours = max(get_context_len(model), min_lookback_hours)
    sample = ForecastSamples(
        *(
            EvalForecast(val, fc, lookback_hours, EvalMetric.from_row(fc_m), future_cov=future_cov)
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
    data_transformers: DataTransformers | None,
    random_state: int | None,
    tz: str | tzinfo,
    run_ts_delta: pd.Timedelta,
    month_filter: tuple[int, int] | None,
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
    metric_df = join_start_end(metric_df, hf_df)

    return get_median_and_samples(hf, metric_df, metric, month_filter), hf_df


def get_median_and_samples(
    hf: list[TimeSeries],
    metric_df: pd.DataFrame,
    metric: MetricType = "MAE",
    month_filter: tuple[int, int] | None = None,
) -> tuple[
    EvalMetric, tuple[tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray], tuple[TimeSeries, np.ndarray]]
]:
    """Get aggregated median metrics and best, worst and median forecast samples."""
    assert len(hf) == metric_df.shape[0], "historical forecast don't match metric df"

    if month_filter:
        # only look at forecasts that start and end within the season
        month_start, month_end = month_filter
        month_idx = (metric_df["start"].dt.month >= month_start) & (metric_df["end"].dt.month <= month_end)
        metric_df = metric_df[month_idx]
        hf = cast(list[TimeSeries], np.array(hf)[month_idx].tolist())

    argsort = metric_df[metric].argsort()
    # using len // 2 results in taking the worst of the 2 median ones if n is even (pessimistic)
    best_i, worst_i, med_i = argsort.iat[0], argsort.iat[-1], argsort.iat[len(argsort) // 2]

    # only need metric columns from now on (in expected order!)
    metric_df = metric_df[metric_names]
    # interestingly median returns an ndarray, std returns a series when using np funcs
    metrics_median = np.median(metric_df, axis=0).astype(float)
    metrics_std = np.std(metric_df, axis=0).astype(float)
    assert isinstance(metrics_median, np.ndarray) and isinstance(metrics_std, pd.Series)
    agg_metrics = EvalMetric.from_row(metrics_median, metrics_std.to_numpy())

    return (
        agg_metrics,
        (
            # forecast with corresponding row of metrics
            (hf[med_i], metric_df.iloc[med_i].to_numpy()),
            (hf[best_i], metric_df.iloc[best_i].to_numpy()),
            (hf[worst_i], metric_df.iloc[worst_i].to_numpy()),
        ),
    )


def hf_to_table(
    hf: list[TimeSeries], tz: str | tzinfo, run_ts_delta: pd.Timedelta = pd.Timedelta(1, "s")
) -> pd.DataFrame:
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

    if "actual" not in true_target.columns:
        non_time_cols = [c for c in true_target.columns if c not in ("time", "_time")]
        assert len(non_time_cols) == 1, (
            f"more than 1 non-time column in true target data! cannot handle: {non_time_cols}"
        )
        true_target = true_target.rename(columns={non_time_cols[0]: "actual"})

    assert "actual" in true_target.columns, f"'actual' column missing in true df (has {true_target.columns})"

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


def join_start_end(metric_df: pd.DataFrame, hf_df: pd.DataFrame) -> pd.DataFrame:
    """Get the start and end times per historical forecast and join them to another df by run_ts."""
    min_max_time = hf_df["time"].groupby(hf_df["run_ts"]).agg(["min", "max"])  # pyright: ignore[reportUnknownMemberType]
    return metric_df.join(min_max_time.rename(columns=dict(min="start", max="end")), on="run_ts")
