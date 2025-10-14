# TODO write metric inspired by darts MAE and AE which calculates the difference between the predicted daily maximum and the actual daily maximum.
# first try without the decorators but you probably need them, not sure what you need to watch out for with numpy for example.
# you should probably transform all points into their daily diff so that half a day gets weighted less when you then average over everything.
# this should also help avoid dimensionality problems that might violate darts expectations of metric functions.
# TODO unlike MAE and RMSE, reuse a single impl that calculates the peak diffs. Maybe even add a small layer of caching since often
# we won't just have mean absolute daily peak diff but also root mean square daily peak diff.

from collections.abc import Sequence
from datetime import tzinfo
from typing import Callable, Optional, Union

import darts
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.metrics.utils import (
    METRIC_OUTPUT_TYPE,
    TIME_AX,
    _get_values_or_raise,
    _get_wrapped_metric,
    multi_ts_support,
    multivariate_support,
)

SHORTEST_MONTH = pd.Timedelta(days=28)


def _add_day_attribute(ts: TimeSeries, tz: str | tzinfo | None = None) -> TimeSeries:
    """Add a 'day' attribute to the time series. Works with stochastic series."""
    attribute = "day"

    if ts.is_deterministic:
        # simple case
        # typing bug in darts: https://github.com/unit8co/darts/issues/2926
        return ts.add_datetime_attribute(attribute, tz=tz)  # pyright: ignore[reportArgumentType]

    ts_det = ts.with_values(ts.values(sample=0))  # take only first sample
    # add_datetime_attribute works now as it's deterministic
    with_date = ts_det.add_datetime_attribute(attribute, tz=tz)  # pyright: ignore[reportArgumentType]
    # must create a ts that matches in sample dimension to add as a component
    day_vals = with_date[attribute]
    day_full = darts.concatenate(
        [day_vals] * ts.n_samples, axis="sample", ignore_time_axis=True, ignore_static_covariates=True
    )

    return ts.concatenate(day_full, axis="component")


@multi_ts_support
@multivariate_support
def dmd(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    tz: str | tzinfo | None = None,
) -> METRIC_OUTPUT_TYPE:
    # signature copied from ae with tz added
    """Daily Maximum Difference (DMD)

    Difference between the actual daily peak and the predicted daily peak. Day border is decided by
    the specified timezone (tz), or the underlying, timezone-naive data if None. In backtest or similar, the timezone
    must be set via metric_kwargs.
    Dimensions are kept and the daily difference is replicated across all time steps for the entire day.
    This ensures consistency with other metrics and correct weighting when taking the mean (partial days should contribute less).
    """
    assert isinstance(actual_series, TimeSeries), "actual_series is not a single TimeSeries, decorator fail?"
    assert isinstance(pred_series, TimeSeries), "pred_series is not a single TimeSeries, decorator fail?"
    assert intersect, "Why and where would intersect ever be false??"
    shorter_dur = min(actual_series.duration, pred_series.duration)

    assert isinstance(shorter_dur, pd.Timedelta)
    if shorter_dur >= SHORTEST_MONTH:
        # the way we group with np.unique only works if the days are consecutive so 30,30,31,31,01,01,02,02 will work
        # but 30,01,30 wouldn't work (unordered) and 31,01,...,30,31 (>= one month) also wouldn't work AFAIK -> raise.
        # since we're only interested in the intersection to calculate the metric, we can just validate the shorter seq
        raise ValueError("Metric currently only supports slices shorter than a month because I was lazy.")

    # add the day as a component to group by when doing daily calculations on numpy array
    actual_series = _add_day_attribute(actual_series, tz)
    pred_series = _add_day_attribute(pred_series, tz)

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        # IIRC remove_nan_union should be set to true when you do calculations with just one of the two like with
        # the relative/scaled metrics that for example devide by the mean of y_true. In that case, it's important that
        # both series have NaN values in the same places otherwise gaps will influence the metric. For metrics that only
        # operate on both series together, e.g. subtracting one from the other can leave this on false, since X - NaN
        # is always NaN so you don't need to make the extra effort to align all the gaps.
        # In this metric, it must be set to true because the daily calculation is done per series, so gaps could influence
        # the maximum in one series but not the other, which we dont want! I hope I understood this correctly..
        remove_nan_union=True,
        # not sure how darts does this internally because the types don't match here but it seems to work..
        q=q,  # pyright: ignore[reportArgumentType]
    )

    # day component is added at end, so use -1 to refer to the last component
    # could also use pandas but I imagine it's slower, despite loop (not tested at all smile)
    def _get_max(arr: np.ndarray):
        # https://stackoverflow.com/a/43094244
        # only works because 'day' is guaranteed to be ordered (not sorted, but unique values will be after each other)
        split_idx = np.unique(arr[:, -1, :], return_index=True)[1][1:]
        days = np.split(arr[:, :-1, :], split_idx)

        # instead of this loop, should be able to use np.repeat (see below)
        max_days = []
        for day in days:
            # max over time (= per component and sample, but should be deterministic here)
            max_day = np.nanmax(day, axis=TIME_AX)
            max_days.append(np.full_like(day, max_day))

        # something like this should work too, but np.stack can't handle differently sized arrays (days)
        # days_stack = np.stack(days)
        # days_max = np.nanmax(days_stack, 1)
        # instead of concat and diff, could use np.unique(return_counts=True)
        # split_idx_full = np.concatenate([[0], split_idx, [arr.shape[0]]])
        # counts = np.diff(split_idx_full)
        # return np.repeat(days_max, counts, axis=0)

        return np.concat(max_days)

    max_true = _get_max(y_true)
    max_pred = _get_max(y_pred)

    return max_true - max_pred


@multi_ts_support
@multivariate_support
def admd(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    tz: str | tzinfo | None = None,
) -> METRIC_OUTPUT_TYPE:
    """Absolute Daily Maximum Difference (ADMD)

    Absolute difference between the actual daily peak and the predicted daily peak. Day border is decided by
    the specified timezone (tz), or the underlying, timezone-naive data if None. In backtest or similar, the timezone
    must be set via metric_kwargs.
    Dimensions are kept and the daily difference is replicated across all time steps for the entire day.
    This ensures consistency with other metrics and correct weighting when taking the mean (partial days should contribute less).
    """
    return np.abs(
        _get_wrapped_metric(dmd)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            tz=tz,
        ),
    )


@multi_ts_support
@multivariate_support
def sdmd(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    tz: str | tzinfo | None = None,
) -> METRIC_OUTPUT_TYPE:
    """Squared Daily Maximum Difference (SDMD)

    Squared difference between the actual daily peak and the predicted daily peak. Day border is decided by
    the specified timezone (tz), or the underlying, timezone-naive data if None. In backtest or similar, the timezone
    must be set via metric_kwargs.
    Dimensions are kept and the daily difference is replicated across all time steps for the entire day.
    This ensures consistency with other metrics and correct weighting when taking the mean (partial days should contribute less).
    """
    return np.power(
        _get_wrapped_metric(dmd)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            tz=tz,
        ),
        2,
    )


@multi_ts_support
@multivariate_support
def madmd(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    tz: str | tzinfo | None = None,
) -> METRIC_OUTPUT_TYPE:
    """Mean Absolute Daily Maximum Difference (MADMD)

    Mean absolute difference between the actual daily peak and the predicted daily peak over all days,
    implicitly weighted by day length (important for partial days).
    Day border is decided by the specified timezone (tz), or the underlying, timezone-naive data if None.
    In backtest or similar, the timezone must be set via metric_kwargs.
    """
    return np.nanmean(
        _get_wrapped_metric(admd)(
            actual_series,
            pred_series,
            intersect,
            q=q,
            tz=tz,
        ),
        axis=TIME_AX,
    )


@multi_ts_support
@multivariate_support
def rmsdmd(
    actual_series: Union[TimeSeries, Sequence[TimeSeries]],
    pred_series: Union[TimeSeries, Sequence[TimeSeries]],
    intersect: bool = True,
    *,
    q: Optional[Union[float, list[float], tuple[np.ndarray, pd.Index]]] = None,
    time_reduction: Optional[Callable[..., np.ndarray]] = None,
    component_reduction: Optional[Callable[[np.ndarray], float]] = np.nanmean,
    series_reduction: Optional[Callable[[np.ndarray], Union[float, np.ndarray]]] = None,
    n_jobs: int = 1,
    verbose: bool = False,
    tz: str | tzinfo | None = None,
) -> METRIC_OUTPUT_TYPE:
    """Root Mean Squared Daily Maximum Difference (RMSDMD)

    Root of mean squared difference between the actual daily peak and the predicted daily peak over all days,
    implicitly weighted by day length (important for partial days).
    Day border is decided by the specified timezone (tz), or the underlying, timezone-naive data if None.
    In backtest or similar, the timezone must be set via metric_kwargs.
    """
    return np.sqrt(
        np.nanmean(
            _get_wrapped_metric(sdmd)(
                actual_series,
                pred_series,
                intersect,
                q=q,
                tz=tz,
            ),
            axis=TIME_AX,
        )
    )
