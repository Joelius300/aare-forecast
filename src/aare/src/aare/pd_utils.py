from collections.abc import Sequence
import functools
import logging
from datetime import tzinfo, datetime
from typing import cast, overload, Callable

import pandas as pd

from aare.constants import TIME

logger = logging.getLogger(__name__)


def _median_filler(df: pd.DataFrame, limit: int):
    n = limit + 1
    return df.rolling(window=n, min_periods=1, center=True).median()


MEDIAN_METHOD_NAME = "median"


@overload
def fill_with_hard_limit(
    df_or_series: pd.DataFrame,
    limit: int,
    fill_func: str | Callable[[pd.DataFrame, int], pd.DataFrame] = "interpolate",
    method: str | None = None,
    columns: list[str] | None = None,
    add_was_filled: bool = False,
    **fill_func_kwargs,
) -> pd.DataFrame:
    pass


@overload
def fill_with_hard_limit(
    df_or_series: pd.Series,
    limit: int,
    fill_func: str | Callable[[pd.DataFrame, int], pd.DataFrame] = "interpolate",
    method: str | None = None,
    columns: list[str] | None = None,
    add_was_filled: bool = False,
    **fill_func_kwargs,
) -> pd.Series:
    pass


def fill_with_hard_limit(
    df_or_series: pd.DataFrame | pd.Series,
    limit: int,
    fill_func: str | Callable[[pd.DataFrame, int], pd.DataFrame] = "interpolate",
    method: str | None = None,
    columns: list[str] | None = None,
    add_was_filled: bool = False,
    **fill_func_kwargs,
) -> pd.DataFrame | pd.Series:
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
    :param add_was_filled: Add 'filled' indicator column.

    :returns: A filled version of the given df_or_series according
        to the given inputs.
    """
    # Keep things simple, ensure we have a DataFrame.
    if isinstance(df_or_series, pd.Series):
        df = df_or_series.to_frame()
    else:
        df = df_or_series
    assert isinstance(df, pd.DataFrame), "df isn't a DataFrame after check?!"

    to_interp = df[columns] if columns else df
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


def between(df: pd.DataFrame, from_: str | datetime | pd.Timestamp, to_: str | datetime | pd.Timestamp):
    """Returns a boolean mask for a time period selection. Assumes '_time' as time column and falls back to index."""
    if TIME in df.columns:
        return (df[TIME] >= from_) & (df[TIME] < to_)

    return (df.index >= from_) & (df.index < to_)


def ensure_frame(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Transform a pandas series/dataframe into a dataframe, if needed."""
    if isinstance(df, pd.Series):
        return df.to_frame()
    return df


def relocalize_times(col: pd.Series, tz: str | tzinfo):
    """Assume naive timestamps are UTC, then convert to the specified timezone. Do nothing if already aware."""
    if col.dt.tz is None:
        return col.dt.tz_localize("UTC").dt.tz_convert(tz)

    assert (isinstance(tz, tzinfo) and col.dt.tz == tz) or (isinstance(tz, str) and col.dt.tz.tzname(None) == tz), (
        f"Times are already aware but the timezone is {col.dt.tz} instead of the requested {tz}!"
    )

    return col


def join_many(*dfs: pd.DataFrame, on: str | Sequence[str]) -> pd.DataFrame:
    """Outer join many dataframes together by (a) common column(s)."""
    return functools.reduce(lambda left, right: pd.merge(left, right, on=on, how="outer"), dfs)
