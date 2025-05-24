# TODO: This module will need a big refactor, since this was just for the temperature feature and we want
#  to transition to the feature class without breaking all the old notebooks etc.
from typing import cast, Optional

from deprecated import deprecated
import numpy as np
import pandas as pd
from darts import TimeSeries

from aare.constants import TEMP, TIME
from aare.params import read_params
from aare.utils import between, fill_with_hard_limit, to_ts


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # resample to add nan points where data is missing. also removes the trailing data point if 18:00 and 18:55 for example.
    # doing this manually gives a bit more control and avoid having to send this data over the air from the influx server.
    return df.set_index(TIME).resample(freq).first().reset_index(TIME)


def resample(df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
    """Resample the dataframe to the (in the params.yaml) specified frequency/resolution."""
    if freq is None:
        freq = read_params()["general"]["frequency"]

    return _resample(df, freq)


@deprecated(reason="Move faulty periods into feature class")
def remove_faulty_periods_aare_temp(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[between(df, "2003-01-28", "2003-03-03"), TEMP] = np.nan

    return df


def remove_outliers(
    df: pd.DataFrame, low_cutoff: float, high_cutoff: float, diff_threshold: float, col=TEMP
) -> pd.DataFrame:
    df = df.copy()
    orig_cols = df.columns

    # eliminate all data points outside valid bound
    df.loc[(df[col] <= low_cutoff) | (df[col] >= high_cutoff), col] = np.nan

    # One variant of outlier is at the start and end of measurement, so [NaN, outlier, normal measurement, ...] or reverse.
    # This is a common pattern in industry sensor data measurements at least from my experience.
    # A slight variation of this first variant is the case when the prev or next measurement is exactly 0 instead of NaN.
    # Another is a random drop or spike so [normal, outlier, normal]. These have both diffs above threshold.
    # All of these variants appear in the data (see EDA).

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

    return cast(pd.DataFrame, df[orig_cols])


@deprecated(reason="Work with WaterTempBern feature")
def remove_outliers_aare_temp(df: pd.DataFrame, col=TEMP) -> pd.DataFrame:
    params = read_params()["cleanup"]

    return remove_outliers(df, params["low_cutoff"], params["high_cutoff"], params["diff_threshold"], col)


def interpolate_continuous(
    df: pd.DataFrame,
    linear_gap_bound: int,
    cubic_gap_bound: int,
    drop_filled: bool,
    columns: str | list[str] | None = TEMP,
):
    df = df.copy()
    if columns is None:
        columns = list(df.columns)
    elif isinstance(columns, str):
        columns = [columns]

    # we NEVER want to interpolate the time
    if TIME in columns:
        columns.remove(TIME)

    col_filled = [c + "_filled" for c in columns]
    df_i = fill_with_hard_limit(df, limit=linear_gap_bound, columns=columns, add_was_filled=True)
    df[columns] = df_i[columns]
    df["filled"] = cast(pd.Series, df_i[col_filled].any(axis=1, bool_only=True)).map({False: "none", True: "linear"})
    df_i = fill_with_hard_limit(df, method="cubic", limit=cubic_gap_bound, columns=columns, add_was_filled=True)
    df[columns] = df_i[columns]
    df.loc[df_i[col_filled].any(axis=1, bool_only=True), "filled"] = "cubic"

    if drop_filled:
        # still populating first for debugging purposed
        df = df.drop("filled", axis="columns")

    return df


@deprecated(reason="Work with WaterTempBern feature")
def interpolate_aare_temp(df: pd.DataFrame, drop_filled=False, columns: str | list[str] | None = TEMP) -> pd.DataFrame:
    """
    Interpolate the temperature column according to the configured options, or more columns.
    Set columns to None for all columns, otherwise it will only to 'temperature'.

    WARNING: The options used are tuned for the water temperature and nothing else. Using it for other variables
    might result in misleading data!!!
    """
    params = read_params()["interpolate"]

    return interpolate_continuous(df, params["linear_gap_bound"], params["cubic_gap_bound"], drop_filled, columns)


@deprecated(reason="Work with WaterTempBern feature")
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
