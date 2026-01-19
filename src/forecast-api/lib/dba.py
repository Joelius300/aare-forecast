from contextlib import asynccontextmanager
from datetime import datetime, timedelta, tzinfo
from typing import cast, Any

import pandas as pd
from psycopg import AsyncConnection
from psycopg.rows import dict_row, TupleRow
from psycopg_pool import AsyncConnectionPool

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


def init_db_pool(connection_string: str) -> AsyncConnectionPool:
    # fixed defaults, no params need atm
    # TODO consolidate with init_db_pool of service into aare-timescale package
    return AsyncConnectionPool(
        connection_string,
        open=False,
        min_size=1,  # keep one connection open at all times
        max_size=4,
        num_workers=1,
        # shouldn't need more workers to manage those connections (big default on min_size, num_workers, ..)
        # kwargs are passed to the connection
        # prepare every query the first time it's executed -> not sure if this works correctly with copy
        kwargs=dict(prepare_threshold=0),
    )
