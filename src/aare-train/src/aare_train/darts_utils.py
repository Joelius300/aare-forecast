import logging
from collections.abc import Sequence
from typing import cast, overload

import darts
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel, GlobalForecastingModel
from darts.utils.ts_utils import retain_period_common_to_all

from aare.constants import TIME
from aare_train.compat.types import ExtremeLags
from aare_train.utils import ensure_frame

logger = logging.getLogger(__name__)


def to_ts(df: pd.DataFrame | pd.Series, col: str | list[str] | None = None, freq: str | None = None) -> TimeSeries:
    """
    Transforms a dataframe into a darts TimeSeries using the predefined TIME column (or index).
    Remove all time zone information.

    Can optionally pass a frequency, otherwise it will try to infer.
    """
    tdf: pd.DataFrame
    if isinstance(df, pd.Series):
        tdf = df.to_frame()
    elif TIME in df.columns:
        tdf = df.set_index(TIME)
    else:
        tdf = df

    if col:
        tdf = ensure_frame(tdf[col])

    assert isinstance(tdf.index, pd.DatetimeIndex), "Index should now be a time index"
    index = cast(pd.DatetimeIndex, tdf.index)
    # turn it into timezone-naive timestamps because that's what darts wants.
    # all the data is in UTC anyway, so a conversion is necessary on display no matter what.
    # TODO check that _if_ the data has a timezone, it's UTC
    index = index.tz_localize(None)

    if index.freq is None:
        if freq is None:
            inf_freq = pd.infer_freq(index)
            if inf_freq is None:
                raise ValueError("Could not infer frequency from data.")
            freq = inf_freq
        # well it works, what do you want me to do pyright?
        index.freq = freq  # pyright: ignore [reportAttributeAccessIssue]
    else:
        if freq != index.freq:
            logger.warning(f"Explicitly passed freq '{freq}', but series already has frequency '{index.freq}'")

    tdf.index = index

    ts = TimeSeries.from_dataframe(tdf, freq=cast(str | int, index.freq))
    # darts only supports 32 and 64 sadly. No need for high 64 precision.
    ts = ts.astype(np.float32)  # pyright: ignore [reportArgumentType]

    return ts


def get_context_len(model: ForecastingModel) -> int:
    """Get the context length of a forecasting model"""
    # written before I realized that extreme_lags[0] should equal the context length, but now incorporated
    extreme_lags = model.extreme_lags
    abs_min_target_lag = abs(extreme_lags[0]) if extreme_lags[0] is not None else None
    if hasattr(model, "context_length"):
        context_len = model.context_length  # pyright: ignore [reportAttributeAccessIssue, reportUnknownMemberType]
    elif hasattr(model, "input_chunk_length"):
        context_len = model.input_chunk_length  # pyright: ignore [reportAttributeAccessIssue, reportUnknownMemberType]
    else:
        context_len = abs_min_target_lag

    if context_len is None:
        raise ValueError("Could not determine context_len of model")

    assert isinstance(context_len, int), "context_len is not int"

    assert abs_min_target_lag is None or context_len == abs_min_target_lag, (
        "Context Length must be indicated by extreme_lags[0]"
    )

    return context_len


def get_data_stats(train_target_subs: list[TimeSeries], val_target_subs: list[TimeSeries]):
    """Get some train/val data stats for logging."""
    train_lens = [len(x) for x in train_target_subs]
    val_lens = [len(x) for x in val_target_subs]

    return {
        "train_lens": train_lens,
        "train_len_total": sum(train_lens),
        "train_n_subs": len(train_lens),
        "val_lens": val_lens,
        "val_len_total": sum(val_lens),
        "val_n_subs": len(val_lens),
        "val_split": sum(val_lens) / (sum(val_lens) + sum(train_lens)),
    }


def trunc_common(*ts: TimeSeries):
    """
    Truncate a set of time series with different components to the longest common continuous slice (no NaN).
    Will return in the same order so x, y, z = trunc_common(x, y, z).
    Make sure that there are no duplicate keys between the components.
    """
    # incorrect typing in darts, retain_period_common_to_all could take any iterable
    tss = list(ts)
    tss = retain_period_common_to_all(tss)  # first truncate to the common time slice (regardless of nan)
    full = darts.concatenate(tss, axis="component")  # then concat all of the components together
    longest = full.longest_contiguous_slice(mode="any")  # then slice and only keep the longest period without nan

    # then reconstruct the individual series
    return tuple(
        [longest[ts.components.to_list()] for ts in tss]  # pyright: ignore[reportArgumentType,reportCallIssue]
    )


@overload
def exclude_short_series(series: Sequence[TimeSeries], min_len: int) -> list[TimeSeries]:
    pass


@overload
def exclude_short_series(series: None, min_len: int) -> None:
    pass


def exclude_short_series(series: Sequence[TimeSeries] | None, min_len: int) -> list[TimeSeries] | None:
    """Create a new list only containing series that are at least min_len long."""
    return [ts for ts in series if len(ts) >= min_len] if series is not None else None


def model_needs_fc(model: GlobalForecastingModel):
    """
    Get whether the model is using future covariates per extreme_lags.
    model.uses_future_covariates is only set after fitting.
    Returns true with a warning if future covariates are required but the lags are only in the past.
    """
    lags: ExtremeLags = model.extreme_lags
    max_fut_lag = lags[5]

    # if the max future lag is none, no future covariates are used at all
    if max_fut_lag is None:
        return False

    if max_fut_lag < 0:
        logger.warning("Maximum future lag is less than 0, so you could use past covariates instead.")

    return True


def get_model_min_len(model: GlobalForecastingModel):
    """
    Get the minimum required length of a training series so that darts has all required
    past data plus all the true future values required to calculate a loss.
    Darts already does something like this internally, so often times it doesn't seem necessary to filter beforehand,
    but apparently it can still fail for some reason, so we have to manually filter first.
    """
    from darts.models import RNNModel

    if isinstance(model, RNNModel):
        return model.training_length

    # for now this seems to work, but I think it would actually need to be something like below, right??
    return abs(model.extreme_lags[0] or 0)
    # return max(abs(model.extreme_lags[0] or 0), abs(model.extreme_lags[2] or 0), abs(model.extreme_lags[4] or 0)) + max(
    #     model.extreme_lags[1] or 0, model.extreme_lags[3] or 0, model.extreme_lags[5] or 0
    # )
