from datetime import datetime
from typing import Optional, Literal

import numpy as np
import pandas as pd
from darts import TimeSeries
from typing_extensions import deprecated

from aare.constants import TEMP, TIME
from aare.params import read_params
from aare.utils import between, fill_with_hard_limit, to_ts


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # resample to add nan points where data is missing. also removes the trailing data point if 18:00 and 18:55 for example.
    # doing this manually gives a bit more control and avoid having to send this data over the air from the influx server.
    return df.set_index(TIME).resample(freq).first().reset_index(TIME)


def resample(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
    """
    Resample the dataframe to the (in the params.yaml) specified frequency/resolution.
    It takes the first and labels the first data point (left). So 08:00 and 08:15 will be
    turned into the single 08:00 point and 08:15 is fully discarded. Does not do any mean
    aggregation or anything of the sorts.
    """
    if freq is None:
        freq = read_params()["general"]["frequency"]

    return _resample(df, freq)


def remove_period(
    df: pd.DataFrame, from_: str | datetime | pd.Timestamp, to_: str | datetime | pd.Timestamp, col: str
) -> None:
    """Set a specific time period in the dataframe to nan."""
    df.loc[between(df, from_, to_), col] = np.nan


def remove_outliers(
    df: pd.DataFrame, low_cutoff: float, high_cutoff: float, diff_threshold: float | None = None, col=TEMP
) -> pd.DataFrame:
    """
    Eliminates (sets to nan) values out of a specific range or with a larger diff than the specified threshold.
    Cutoffs and threshold are INCLUSIVE and data will only be excluded when its higher or lower, not equal.
    """
    df = df.copy()
    orig_cols = df.columns

    # eliminate all data points outside valid bound
    df.loc[(df[col] < low_cutoff) | (df[col] > high_cutoff), col] = np.nan

    # One variant of outlier is at the start and end of measurement, so [NaN, outlier, normal measurement, ...] or reverse.
    # This is a common pattern in industry sensor data measurements at least from my experience.
    # A slight variation of this first variant is the case when the prev or next measurement is exactly 0 instead of NaN.
    # Another is a random drop or spike so [normal, outlier, normal]. These have both diffs above threshold.
    # All of these variants appear in the data (see EDA).

    if diff_threshold is not None:
        df["_diff_to_prev"] = df[col].diff().abs()
        df["_diff_to_next"] = df[col].diff(-1).abs()

        # variant 1a
        df.loc[
            (df["_diff_to_prev"].isna() | (df["_diff_to_prev"] == 0)) & (df["_diff_to_next"] > diff_threshold),
            col,
        ] = np.nan
        # variant 1b
        df.loc[
            (df["_diff_to_next"].isna() | (df["_diff_to_next"] == 0)) & (df["_diff_to_prev"] > diff_threshold),
            col,
        ] = np.nan
        # variant 2
        df.loc[
            (df["_diff_to_prev"] > diff_threshold) & (df["_diff_to_next"] > diff_threshold),
            col,
        ] = np.nan

    return df[orig_cols]


def _interpolate_continuous(
    df: pd.DataFrame,
    gap_bound: int,
    method: str | Literal["linear", "cubic", "median"],
    columns: list[str],
):
    """
    Interpolate gaps in specified columns with a specific method, but only up to a specific gap size.
    Populates a 'filled' column with the method that was used.
    """
    col_filled = [c + "_filled" for c in columns]
    df_i = fill_with_hard_limit(df, limit=gap_bound, method=method, columns=columns, add_was_filled=True)
    df[columns] = df_i[columns]
    df.loc[df_i[col_filled].any(axis=1, bool_only=True), "filled"] = method

    return df


def interpolate_continuous(
    df: pd.DataFrame,
    linear_gap_bound: Optional[int],
    cubic_gap_bound: Optional[int],
    median_gap_bound: Optional[int],
    drop_filled: bool,
    columns: str | list[str] | None = TEMP,
) -> pd.DataFrame:
    """
    Interpolate gaps with different methods up to specified gap sizes.
    Setting linear_gap_bound to 10 will fill all gaps from size 1 to size 10 with linear interpolation.
    If another method is set to something lower, e.g. 5, then all gaps up to 5 will already be filled
    and linear only does 6 through 10.
    Populates a 'filled' column with the method that was used for each data point (or none if not interpolated).
    """
    df = df.copy()
    if columns is None:
        columns = list(df.columns)
    elif isinstance(columns, str):
        columns = [columns]

    # we NEVER want to interpolate the time
    if TIME in columns:
        columns.remove(TIME)

    methods = [
        (linear_gap_bound, "linear") if linear_gap_bound else None,
        (cubic_gap_bound, "cubic") if cubic_gap_bound else None,
        (median_gap_bound, "median") if median_gap_bound else None,
    ]
    methods = [m for m in methods if m is not None]
    if len(methods) == 0:
        raise ValueError("Must specify the gap bound for at least one method.")

    bounds = [m[0] for m in methods]
    if len(bounds) != len(set(bounds)):
        raise ValueError("Cannot specify the same size for multiple interpolation methods.")

    df["filled"] = "none"

    for gap_bound, method in sorted(methods, key=lambda t: t[0]):
        df = _interpolate_continuous(df, gap_bound, method, columns)

    if drop_filled:
        # still populating first for debugging purposed, then drop before returning
        df = df.drop("filled", axis="columns")

    return df


@deprecated("Move faulty periods into feature class")
def remove_faulty_periods_aare_temp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[between(df, "2003-01-28", "2003-03-03"), TEMP] = np.nan

    return df


@deprecated("Work with WaterTempBern feature")
def remove_outliers_aare_temp(df: pd.DataFrame, col=TEMP) -> pd.DataFrame:
    params = read_params()["features"].get("temp_bern", {}).get("outliers")
    if params is None:
        raise ValueError("Outlier config of temp_bern is not configured correctly.")

    return remove_outliers(df, params["low_cutoff"], params["high_cutoff"], params["diff_threshold"], col)


@deprecated("Work with WaterTempBern feature")
def interpolate_aare_temp(df: pd.DataFrame, drop_filled=False, columns: str | list[str] | None = TEMP) -> pd.DataFrame:
    """
    Interpolate the temperature column according to the configured options, or more columns.
    Set columns to None for all columns, otherwise it will only to 'temperature'.

    WARNING: The options used are tuned for the water temperature and nothing else. Using it for other variables
    might result in misleading data!!!
    """
    params = read_params()["features"].get("temp_bern", {}).get("interpolate")
    if params is None:
        raise ValueError("Outlier config of temp_bern is not configured correctly.")

    return interpolate_continuous(
        df,
        params.get("linear_gap_bound"),
        params.get("cubic_gap_bound"),
        params.get("median_gap_bound"),
        drop_filled,
        columns,
    )


@deprecated("Work with WaterTempBern feature")
def prepare_ts_aare_temp(raw: pd.DataFrame) -> TimeSeries:
    """
    Run all the preparation steps on the raw data and return a clean TimeSeries.

    WARNING: Might still contain gaps and must be split with extract_subseries.
    """
    ts = resample(raw)
    ts = remove_faulty_periods_aare_temp(ts)
    ts = remove_outliers_aare_temp(ts)
    ts = interpolate_aare_temp(ts, drop_filled=True)
    ts = to_ts(ts)

    return ts
