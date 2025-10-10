# TODO write metric inspired by darts MAE and AE which calculates the difference between the predicted daily maximum and the actual daily maximum.
# first try without the decorators but you probably need them, not sure what you need to watch out for with numpy for example.
# you should probably transform all points into their daily diff so that half a day gets weighted less when you then average over everything.
# this should also help avoid dimensionality problems that might violate darts expectations of metric functions.

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.dataprocessing import dtw
from darts.logging import get_logger, raise_log
from darts.metrics.utils import (
    METRIC_OUTPUT_TYPE,
    SMPL_AX,
    TIME_AX,
    _compute_score,
    _confusion_matrix,
    _get_error_scale,
    _get_quantile_intervals,
    _get_values_or_raise,
    _get_wrapped_metric,
    _LabelReduction,
    classification_support,
    interval_support,
    multi_ts_support,
    multivariate_support,
)

# @multi_ts_support
# @multivariate_support
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
) -> METRIC_OUTPUT_TYPE:
    # copied from ae
    """Absolute Daily Maximum Difference.

    TODO DODODO

    """

    y_true, y_pred = _get_values_or_raise(
        actual_series,
        pred_series,
        intersect,
        remove_nan_union=False,
        q=q,
    )
    return np.abs(y_true - y_pred)