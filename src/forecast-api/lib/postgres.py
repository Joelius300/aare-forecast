from io import BytesIO
from typing import Optional, LiteralString

import pandas as pd
import psycopg
from psycopg.abc import Params
from psycopg.rows import dict_row
from psycopg import sql
from datetime import datetime, timedelta

from lib.dto import ModelInfo


# duplicate code, I'm a (lazy) sinner
async def copy_to_df(
    conn: psycopg.AsyncConnection, query: LiteralString | sql.SQL | sql.Composed, params: Optional[Params] = None
) -> pd.DataFrame:
    """Copy a Timescale query into a pandas DataFrame."""
    if isinstance(query, str):
        query = sql.SQL(query)
    async with conn.cursor() as cur:
        with BytesIO() as bio:
            async with cur.copy(
                sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query), params
            ) as copy:
                async for data in copy:
                    bio.write(data)
            bio.seek(0)

            return pd.read_csv(bio)


async def select_forecasts(
    conn: psycopg.AsyncConnection, at: datetime, lookback: str | timedelta, horizon: int, city: str
) -> pd.DataFrame:
    if isinstance(lookback, str):
        lookback = pd.to_timedelta(lookback).to_pytimedelta()

    if not (city.isascii() and city.isalpha()):
        raise ValueError(f"Not sure how, but an invalid (potentially dangerous) city got through: {city}")
    value_col = f"temp_{city}"

    # precision is up to 6 digits after decimal point, could use TRUNC or ROUND here to avoid but why ¯\_(ツ)_/¯
    query = sql.SQL("""select distinct on (time)
          run_ts,
          time,
          {value_col} as temp
        from forecast
        where run_ts between %(at)s - %(lookback)s and %(at)s
          and time >= %(at)s
        order by time, run_ts desc
        limit %(horizon)s
    """).format(value_col=sql.Identifier(value_col))
    params = dict(at=at, lookback=lookback, horizon=horizon)
    df = await copy_to_df(conn, query, params)

    return df


async def get_model_info(conn: psycopg.AsyncConnection, run_ts: datetime) -> ModelInfo:
    # could also just "join forecast_meta as meta on pred.run_ts=meta.run_ts" in select_forecasts
    async with conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(
            "select model_name as name, model_version as version from forecast_meta where run_ts=%s", [run_ts]
        )
        row = await cur.fetchone()
        assert row is not None, "got none when selecting model, what run_ts did you pass??"

        return ModelInfo.model_validate(row)
