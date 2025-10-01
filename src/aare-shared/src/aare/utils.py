import logging
import os
from pathlib import Path
from typing import Union, Optional, cast, overload, Callable

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel

from aare.constants import TIME

logger = logging.getLogger(__name__)


def find_project_root(raise_not_found=True, allow_env=True) -> Path:
    """Traverse CWD up to the project root and return its path. Take PROJECT_ROOT env if set."""
    if allow_env and (root := os.getenv("PROJECT_ROOT")):
        return Path(root)

    cwd = Path(os.getcwd())
    cur = cwd
    while True:
        # should be enough to identify the project root
        if (cur / ".dvc").is_dir() and (cur / "uv.lock").is_file():
            return cur

        parent = cur.parent

        if parent == cur:
            # reached file system root
            if raise_not_found:
                raise ValueError("Could not find project root!")
            else:
                logger.warning(f"COULD NOT DETERMINE PROJECT ROOT, USING CWD: '{cwd}'")
                return cwd

        cur = parent


# suboptimal that this runs on import, but works and avoids refactoring many things
PROJECT_ROOT = find_project_root(raise_not_found=False)
DATA_FOLDER: Path = PROJECT_ROOT / "data"

METRICS_FOLDER = DATA_FOLDER / "metrics"
FORECAST_SAMPLES_FOLDER = DATA_FOLDER / "forecast_samples"
MODELS_FOLDER = DATA_FOLDER / "models"

OPTUNA_STORE = DATA_FOLDER / "optuna-trials.db"
OPTUNA_STORE_URI = f"sqlite:///{OPTUNA_STORE}"


def _median_filler(df: pd.DataFrame, limit: int):
    n = limit + 1
    return df.rolling(window=n, min_periods=1, center=True).median()


MEDIAN_METHOD_NAME = "median"


@overload
def fill_with_hard_limit(
    df_or_series: pd.DataFrame,
    limit: int,
    fill_func="interpolate",
    method: Optional[str] = None,
    columns: Optional[list[str]] = None,
    add_was_filled=False,
    **fill_func_kwargs,
) -> pd.DataFrame:
    pass


@overload
def fill_with_hard_limit(
    df_or_series: pd.Series,
    limit: int,
    fill_func="interpolate",
    method: Optional[str] = None,
    columns: Optional[list[str]] = None,
    add_was_filled=False,
    **fill_func_kwargs,
) -> pd.Series:
    pass


def fill_with_hard_limit(
    df_or_series: Union[pd.DataFrame, pd.Series],
    limit: int,
    fill_func: str | Callable[[pd.DataFrame, int], pd.DataFrame] = "interpolate",
    method: Optional[str] = None,
    columns: Optional[list[str]] = None,
    add_was_filled=False,
    **fill_func_kwargs,
) -> Union[pd.DataFrame, pd.Series]:
    # adjusted from https://stackoverflow.com/a/66373000/10883465
    """
    The fill methods from Pandas such as ``interpolate`` or ``bfill``
    will fill ``limit`` number of NaNs, even if the total number of
    consecutive NaNs is larger than ``limit``. This function instead
    does not fill any data when the number of consecutive NaNs
    is > ``limit``. ``median`` is also supported; either set fill_func or method.

    Adapted from: https://stackoverflow.com/a/30538371/11052174

    :param df_or_series: DataFrame or Series to perform interpolation
        on.
    :param limit: Maximum number of consecutive NaNs to allow. Any
        occurrences of more consecutive NaNs than ``limit`` will have no
        filling performed.
    :param fill_func: Filling method to use, e.g. 'interpolate',
        'bfill', etc. or a lambda taking 'limit' and potential method and fill_kwargs. 'linear', 'cubic', etc.
        must be specified through the 'method' arg, not here.
    :param method: The 'method' kwarg of the pandas 'interpolate' method (or the specified 'fill_func').
        Only needs to be set if the fill_func takes 'method' as an argument.
    :param columns: Which columns so fill. Defaults to all.
    :param fill_func_kwargs: Keyword arguments to pass to the
        fill_func, in addition to the given limit and method.

    :returns: A filled version of the given df_or_series according
        to the given inputs.
    """
    # Keep things simple, ensure we have a DataFrame.
    if isinstance(df_or_series, pd.Series):
        df = df_or_series.to_frame()
    else:
        df = df_or_series
    assert isinstance(df, pd.DataFrame), "df isn't a DataFrame after check?!"

    to_interp = cast(pd.DataFrame, df[columns] if columns else df)
    columns = list(to_interp.columns)

    # Initialize our mask.
    mask = pd.DataFrame(True, index=to_interp.index, columns=to_interp.columns)

    # Get cumulative sums of consecutive NaNs.
    grp = (to_interp.notnull() != to_interp.shift().notnull()).cumsum()

    # Add columns of ones.
    grp["ones"] = 1

    # Loop through columns and update the mask.
    for col in columns:
        grp_counts = grp.groupby(col)["ones"].transform("count")
        # (grp_counts <= limit) returns a mask for all parts that are shorter or equal to the specified limit.
        # Note that the "parts" are separated at places where it switches from nan to non-nan or vice versa, but
        # the mask includes all parts that are this short, which may contain only nan or no nans at all.
        # To make sure the mask only contains the groups that consist of nans, combine it with to_interp[col].isna().
        # When using combine_first, it wouldn't matter because it only takes values from the interpolated df if the
        # value in the original df is nan (which is obviously only the case in the short parts consisting only of nans).
        mask.loc[:, col] = (grp_counts <= limit) & to_interp[col].isna()

    if "method" in fill_func_kwargs:
        raise ValueError("Don't set 'method' in fill_func_kwargs, use the 'method' param directly.")
    elif method:
        # add method to kwargs, but only if it's not None
        fill_func_kwargs["method"] = method

    if fill_func == MEDIAN_METHOD_NAME or method == MEDIAN_METHOD_NAME:
        # custom moving median implementation
        interpolated = _median_filler(to_interp, limit=limit)
    elif isinstance(fill_func, str):
        interp_method = getattr(to_interp, fill_func)
        interpolated = interp_method(limit=limit, **fill_func_kwargs)
    else:
        # ignore because kwargs aren't supported for Callable
        # noinspection PyArgumentList
        interpolated = fill_func(to_interp, limit, **fill_func_kwargs)

    # only take those parts that were from NaN-only sections shorter than the specified limit
    interpolated = interpolated[mask]

    # put the filled values in the short missing sections of the original
    out = df.combine_first(cast(pd.DataFrame, interpolated))

    # add extra columns to show which values were filled in
    if add_was_filled:
        was_filled_mask = ~to_interp.notnull() & out.notnull()
        for c in columns:
            out[c + "_filled"] = was_filled_mask[c]

    # Be nice to the caller and return a Series if that's what they provided.
    if isinstance(df_or_series, pd.Series) and not add_was_filled:
        # Return a Series.
        return out.loc[:, out.columns[0]]

    return out


def between(df, from_, to_):
    """Returns a boolean mask for a time period selection. Assumes '_time' as time column and falls back to index."""
    if TIME in df.columns:
        return (df[TIME] >= from_) & (df[TIME] < to_)

    return (df.index >= from_) & (df.index < to_)


def ensure_frame(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Transform a pandas series/dataframe into a dataframe, if needed."""
    if isinstance(df, pd.Series):
        return df.to_frame()
    return df


def to_ts(df: pd.DataFrame | pd.Series, col: Optional[str | list[str]] = None, freq=None) -> TimeSeries:
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
        context_len = model.context_length  # pyright: ignore [reportAttributeAccessIssue]
    elif hasattr(model, "input_chunk_length"):
        context_len = model.input_chunk_length  # pyright: ignore [reportAttributeAccessIssue]
    else:
        context_len = abs_min_target_lag

    if context_len is None:
        raise ValueError("Could not determine context_len of model")

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
