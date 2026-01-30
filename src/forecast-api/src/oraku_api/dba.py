from datetime import datetime, timedelta, tzinfo
from typing import cast, LiteralString

import pandas as pd
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from aare.locations import LOC_ALIAS
from aare_timescale.forecasts import select_forecasts
from aare_timescale.postgres import copy_to_df

from oraku_api.dto import ModelInfo


async def _fetch_flow_forecast(
    conn: AsyncConnection, at: datetime, lookback: str | timedelta, horizon: int, city: str
) -> pd.DataFrame:
    """
    Select BAFU flow forecasts made between 'at' and 'at - lookback' for a specific location/city with a specific horizon.
    NOTE: Will rename our estimate of when the forecast was made to run_ts for compatibility, but do the actual
    lookback check on the original run_ts, to make sure it only fails if WE didn't fetch the forecasts in the specified
    lookback (no matter how old the forecast actually is). This is mostly for consistency with the rest and because
    the lookback is designed to reasonably handle forecast service outages (rather than "quality control").
    If forecasts need to be made more often, that's not an issue of the API. Also avoids separate maximum_forecast_age.
    """
    if isinstance(lookback, str):
        lookback = pd.to_timedelta(lookback).to_pytimedelta()

    loc = LOC_ALIAS[city.upper()]["hydro"]
    if not loc:
        raise ValueError(f"Could not determine hydro station id of '{city}'")
    loc = str(loc)

    # IMPORTANT NOTE: the run_ts in the WHERE references the original run_ts (= when it was fetched via our service).
    # The run_ts in the ORDER BY references the renamed one (originally last_updated, which is our estimate of when
    # BAFU's models made this forecast based on the first time in the response).
    # Also note that the result is also sorted by the original run_ts in addition, to make sure that the returned
    # data all originates from a single service run on our side. If we didn't do this, the DISTINCT ON would take
    # the first on the arbitrary row order where only last_updated (renamed to run_ts) is sorted DESC, which would mean
    # that all returned rows have the same last_updated (renamed to run_ts), but not necessarily the same original run_ts.
    query: LiteralString = """select distinct on (time)
          run_ts as _run_ts,
          last_updated as run_ts,
          time,
          flow
        from bafu_flow
        where location = %(loc)s
          and run_ts between %(at)s - %(lookback)s and %(at)s
          and time >= %(at)s
        order by time, run_ts desc, _run_ts desc
        limit %(horizon)s
    """
    params = dict(at=at, loc=loc, lookback=lookback, horizon=horizon)
    df = await copy_to_df(conn, query, params)

    return df


async def fetch_forecast(
    conn: AsyncConnection, variable: str, from_: datetime, horizon: int, city: str, max_age: str | timedelta, tz: tzinfo
) -> tuple[datetime | None, pd.DataFrame]:
    """
    Get a forecast, localize times and extract the run_ts. Returns (None, empty-df) if no forecast was found.
    NOTE: For flow forecasts (external from BAFU), run_ts is actually our estimate of when they made their forecast
    based on the first time in their response (saved as last_updated in the db).
    """
    # regarding downstream compatibility that run_ts here is when we _think_ BAFU made the forecast:
    #  - it's also used as cache key, but that shouldn't matter since my simple tests show that the values are all
    #    the same when last_updated is the same (duh, but you never know, the length is also inconsistent).
    if variable == "temp":
        df = await select_forecasts(conn, from_, max_age, horizon, city)
    elif variable == "flow":
        df = await _fetch_flow_forecast(conn, from_, max_age, horizon, city)
    else:
        raise ValueError(f"Invalid variable '{variable}'")

    if df.empty:
        return None, df

    # if speed is important, it's probably faster to use .dt.strftime() to convert pd.Timestamp directly to str
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(tz).apply(pd.Timestamp.to_pydatetime)

    run_ts_unique = df["run_ts"].unique()
    assert len(run_ts_unique) == 1, "fetched more than one run, currently not supported so it shouldn't happen"
    run_ts = cast(str, run_ts_unique.item())
    run_ts = datetime.fromisoformat(run_ts).astimezone(tz)

    return run_ts, df


async def fetch_model_info(conn: AsyncConnection, run_ts: datetime) -> ModelInfo:
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
