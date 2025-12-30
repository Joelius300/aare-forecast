from datetime import datetime, timedelta

import pandas as pd
from psycopg import AsyncConnection, sql

from aare_timescale.postgres import copy_to_df


# not sure if this should be part of the shared timescale package, but later on we'll want to fetch
# forecasts during training as well, so that is a good argument IMO. We'll see.
async def select_forecasts(
    conn: AsyncConnection, at: datetime, lookback: str | timedelta, horizon: int, city: str
) -> pd.DataFrame:
    """Select forecasts made between 'at' and 'at - lookback' for a specific location/city with a specific horizon."""
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
