# TODO write metric inspired by darts MAE and AE which calculates the difference between the predicted daily maximum and the actual daily maximum.
# first try without the decorators but you probably need them, not sure what you need to watch out for with numpy for example.
# you should probably transform all points into their daily diff so that half a day gets weighted less when you then average over everything.
# this should also help avoid dimensionality problems that might violate darts expectations of metric functions.

from collections.abc import Sequence
from datetime import tzinfo
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.metrics.utils import (
    METRIC_OUTPUT_TYPE,
    _get_values_or_raise,
    multi_ts_support,
    multivariate_support,
)


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
    # signature copied from ae
    """Absolute Daily Maximum Difference.

    Absolute difference between the actual daily peak and the predicted daily peak. Day border is decided by
    the specified timezone (tz), or the underlying, timezone-naive data if None. In backtest or similar, the timezone
    must be set via metric_kwargs.
    Dimensions are kept, so if a day TODO.
    """
    assert isinstance(actual_series, TimeSeries), (
        "actual_series is not a single TimeSeries, how do those decorators work??"
    )
    assert isinstance(pred_series, TimeSeries), "pred_series is not a single TimeSeries, how do those decorators work??"
    assert isinstance(actual_series.duration, pd.Timedelta)
    assert isinstance(pred_series.duration, pd.Timedelta)
    month = pd.Timedelta(days=28)
    if pred_series.duration >= month or actual_series.duration >= month:
        raise ValueError(
            "Metric currently only supports slices shorter than a month because I was lazy in the implementation."
        )

    # add the day as a component to group by when doing daily calculations on numpy array
    # typing bug in darts: https://github.com/unit8co/darts/issues/2926
    actual_series = actual_series.add_datetime_attribute("day", tz=tz)  # pyright: ignore[reportArgumentType]
    pred_series = pred_series.add_datetime_attribute("day", tz=tz)  # pyright: ignore[reportArgumentType]

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
        days = np.split(arr[:, :-1, :], np.unique(arr[:, -1, :], return_index=True)[1][1:])
        max_days = []
        for day in days:
            # max over time (= per component and sample, not sure about implications with those decorators)
            max_day = np.nanmax(day, axis=0)
            max_days.append(np.full_like(day, max_day))

        return np.concat(max_days)

    max_true = _get_max(y_true)
    max_pred = _get_max(y_pred)

    return np.abs(max_true - max_pred)
