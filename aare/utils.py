import logging
from typing import Union, Optional

import pandas as pd
from darts import TimeSeries

from aare.constants import TIME

logger = logging.getLogger(__name__)


def fill_with_hard_limit(
        df_or_series: Union[pd.DataFrame, pd.Series], limit: int,
        fill_method='interpolate',
        columns: Optional[list[str]] = None,
        add_was_filled=False,
        **fill_method_kwargs) -> Union[pd.DataFrame, pd.Series]:
    # adjusted from https://stackoverflow.com/a/66373000/10883465
    """The fill methods from Pandas such as ``interpolate`` or ``bfill``
    will fill ``limit`` number of NaNs, even if the total number of
    consecutive NaNs is larger than ``limit``. This function instead
    does not fill any data when the number of consecutive NaNs
    is > ``limit``.

    Adapted from: https://stackoverflow.com/a/30538371/11052174

    :param df_or_series: DataFrame or Series to perform interpolation
        on.
    :param limit: Maximum number of consecutive NaNs to allow. Any
        occurrences of more consecutive NaNs than ``limit`` will have no
        filling performed.
    :param fill_method: Filling method to use, e.g. 'interpolate',
        'bfill', etc.
    :param columns: Which columns so fill. Defaults to all.
    :param fill_method_kwargs: Keyword arguments to pass to the
        fill_method, in addition to the given limit.

    :returns: A filled version of the given df_or_series according
        to the given inputs.
    """
    # Keep things simple, ensure we have a DataFrame.
    try:
        df = df_or_series.to_frame()
    except AttributeError:
        df = df_or_series

    to_interp = df[columns] if columns else df
    columns = to_interp.columns

    # Initialize our mask.
    mask = pd.DataFrame(True, index=to_interp.index, columns=to_interp.columns)

    # Get cumulative sums of consecutive NaNs.
    grp = (to_interp.notnull() != to_interp.shift().notnull()).cumsum()

    # Add columns of ones.
    grp['ones'] = 1

    # Loop through columns and update the mask.
    for col in columns:
        mask.loc[:, col] = (
                (grp.groupby(col)['ones'].transform('count') <= limit)
                | to_interp[col].notnull()
        )

    # Now, interpolate and use the mask to create NaNs for the larger gaps.
    method = getattr(to_interp[columns], fill_method)
    interpolated = method(limit=limit, **fill_method_kwargs)[mask]

    out = df.copy()
    for c in columns:
        out[c] = interpolated[c]

    if add_was_filled:
        was_filled_mask = (~to_interp.notnull() & out.notnull())
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


def to_ts(df, freq=None):
    """
    Transforms a dataframe into a darts TimeSeries using the predefined TIME column (or index).
    Remove all time zone information.

    Can optionally pass a frequency, otherwise it will try to infer.
    """
    if TIME in df.columns:
        tdf = df.set_index(TIME)
    else:
        tdf = df

    # turn it into timezone-naive timestamps because that's what darts wants.
    # all the data is in UTC anyway, so a conversion is necessary on display no matter what.
    tdf.index = tdf.index.tz_localize(None)

    if tdf.index.freq is None:
        if freq is None:
            inf_freq = pd.infer_freq(tdf.index)
            if inf_freq is None:
                raise ValueError("Could not infer frequency from data.")
            freq = inf_freq
        tdf.index.freq = freq
    else:
        if freq != tdf.index.freq:
            logger.warning(f"Explicitly passed freq '{freq}', but series already has frequency '{tdf.index.freq}'")

    return TimeSeries.from_dataframe(tdf, freq=tdf.index.freq)
