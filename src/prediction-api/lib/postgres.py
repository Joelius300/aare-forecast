from io import BytesIO

import pandas as pd
import psycopg
from psycopg import sql
from datetime import datetime, timedelta


# duplicate code, I'm a (lazy) sinner
async def copy_to_df(conn: psycopg.AsyncConnection, query: sql.SQL | sql.Composed) -> pd.DataFrame:
    """Copy a Timescale query into a pandas DataFrame."""
    async with conn.cursor() as cur:
        with BytesIO() as bio:
            async with cur.copy(sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query)) as copy:
                async for data in copy:
                    bio.write(data)
            bio.seek(0)

            return pd.read_csv(bio)


async def select_predictions(
    conn: psycopg.AsyncConnection, at: datetime, lookback: str | timedelta, horizon: int
) -> pd.DataFrame:
    if isinstance(lookback, str):
        lookback = pd.to_timedelta(lookback).to_pytimedelta()

    query = sql.SQL("""
    select distinct on (time)
      run_ts,
      time,
      temp_bern
    from prediction
    where run_ts between {at} - {lookback} and {at}
      and time >= {at}
    order by time, run_ts desc
    limit {horizon}
    """).format(at=at, lookback=lookback, horizon=horizon)
    df = await copy_to_df(conn, query)

    return df
