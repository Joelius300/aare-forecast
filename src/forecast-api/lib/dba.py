from datetime import datetime, timedelta, tzinfo
from typing import cast

import pandas as pd
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from lib.dto import ModelInfo
from aare_timescale.forecasts import select_forecasts


async def fetch_forecast(
    conn: AsyncConnection, from_: datetime, horizon: int, city: str, max_age: str | timedelta, tz: tzinfo
) -> tuple[datetime | None, pd.DataFrame]:
    """Get a forecast, localize times and extract the run_ts. Returns (None, empty-df) if no forecast was found."""
    df = await select_forecasts(conn, from_, max_age, horizon, city)
    if df.empty:
        return None, df

    # if speed is important, it's probably faster to use .dt.strftime() to convert pd.Timestamp directly to str
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(tz).apply(pd.Timestamp.to_pydatetime)

    run_ts_unique = df["run_ts"].unique()
    assert len(run_ts_unique) == 1, "fetched more than one run, currently not supported so it shouldn't happen"
    run_ts = cast(str, run_ts_unique.item())
    run_ts = datetime.fromisoformat(run_ts).astimezone(tz)

    return run_ts, df


async def get_model_info(conn: AsyncConnection, run_ts: datetime) -> ModelInfo:
    # could also just "join forecast_meta as meta on pred.run_ts=meta.run_ts" in select_forecasts
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "select model_name as name, model_version as version from forecast_meta where run_ts=%s", [run_ts]
        )
        row = await cur.fetchone()
        assert row is not None, "got none when selecting model, what run_ts did you pass??"

        return ModelInfo.model_validate(row)
