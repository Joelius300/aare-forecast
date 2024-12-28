from typing import cast

import numpy as np
import pandas as pd

from aare.constants import TEMP, TIME
from aare.utils import between, fill_with_hard_limit


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    # resample to add nan points where data is missing. also removes the trailing data point if 18:00 and 18:55 for example.
    # doing this manually gives a bit more control and avoid having to send this data over the air from the influx server.
    return df.set_index(TIME).resample(freq).first().reset_index(TIME)


def remove_faulty_periods(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[between(df, "2003-01-28", "2003-03-03"), TEMP] = np.nan

    return df


def _remove_outliers(
    df: pd.DataFrame, low_cutoff: float, high_cutoff: float, diff_threshold: float
) -> pd.DataFrame:
    df = df.copy()
    orig_cols = df.columns

    # eliminate all data points outside valid bound
    df.loc[(df[TEMP] <= low_cutoff) | (df[TEMP] >= high_cutoff), TEMP] = np.nan

    # One variant of outlier is at the start and end of measurement, so [NaN, outlier, normal measurement, ...] or reverse.
    # This is a common pattern in industry sensor data measurements at least from my experience.
    # A slight variation of this first variant is the case when the prev or next measurement is exactly 0 instead of NaN.
    # Another is a random drop or spike so [normal, outlier, normal]. These have both diffs above threshold.
    # All of these variants appear in the data (see EDA).

    df["temp_diff_to_prev"] = df[TEMP].diff().abs()
    df["temp_diff_to_next"] = df[TEMP].diff(-1).abs()

    # variant 1a
    df.loc[
        (df["temp_diff_to_prev"].isna() | (df["temp_diff_to_prev"] == 0))
        & (df["temp_diff_to_next"] > diff_threshold),
        TEMP,
    ] = np.nan
    # variant 1b
    df.loc[
        (df["temp_diff_to_next"].isna() | (df["temp_diff_to_next"] == 0))
        & (df["temp_diff_to_prev"] > diff_threshold),
        TEMP,
    ] = np.nan
    # variant 2
    df.loc[
        (df["temp_diff_to_prev"] > diff_threshold)
        & (df["temp_diff_to_next"] > diff_threshold),
        TEMP,
    ] = np.nan

    return cast(pd.DataFrame, df[orig_cols])


def _interpolate(df: pd.DataFrame, linear_gap_bound: int, cubic_gap_bound: int):
    df = df.copy()

    df_i = fill_with_hard_limit(
        df, limit=linear_gap_bound, columns=[TEMP], add_was_filled=True
    )
    df[TEMP] = df_i[TEMP]
    df["filled"] = cast(pd.Series, df_i[TEMP + "_filled"]).map(
        {False: "none", True: "linear"}
    )

    df_i = fill_with_hard_limit(
        df, method="cubic", limit=cubic_gap_bound, columns=[TEMP], add_was_filled=True
    )
    df[TEMP] = df_i[TEMP]
    df.loc[df_i[TEMP + "_filled"], "filled"] = "cubic"

    return df
