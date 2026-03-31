from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from collections import defaultdict
import logging
from typing import cast

import uvloop
from pandas import DataFrame
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from aare.constants import TIME
from aare_logging.logging import setup_logging
from aare_timescale.postgres import init_db_pool, copy_to_df, upsert_df
from aare_timescale.timescale import make_hypertable
from aare_influx.remote_existenz_store import RemoteExistenzStore
from aare_influx.field_request import FieldRequest
from oraku_influx2pg.args import parse_cli_args

logger = logging.getLogger(__name__)

DEFAULT_SINCE = datetime(2026, 1, 1, tzinfo=UTC)
TABLE_NAME_PREFIX = "mirror_"
EXPECTED_FREQ = "1h"  # not supporting anything else right now


async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-influx2pg")
    logger.info(f"Mirroring the following influx variables to pg: {args.fields}")

    store = RemoteExistenzStore()
    fields = [FieldRequest.from_str(f) for f in args.fields]
    mirrored_at = datetime.now(UTC)

    async with init_db_pool(args.connection_string) as pool:
        grp_measurement_fields: dict[str, list[FieldRequest]] = defaultdict(list)
        for fr in fields:
            assert fr.freq == EXPECTED_FREQ, f"Got unexpected frequency '{fr.freq}' (instead of '{EXPECTED_FREQ}')"
            grp_measurement_fields[fr.measurement].append(fr)

        for measurement, meas_fields in grp_measurement_fields.items():
            table_name = await create_measurement_table(measurement, meas_fields, pool)

            # group fields by location[_orig] since all variables share a row per time and location
            grp_loc_fields: dict[str, list[FieldRequest]] = defaultdict(list)
            for fr in meas_fields:
                grp_loc_fields[str(fr.location_orig).upper()].append(fr)

            # fetch latest mirrored timestamp per location and compute global first time.
            since_per_loc = await get_latest_times(table_name, grp_loc_fields, pool)
            global_since = min(since_per_loc.values())
            logger.info(f"Fetching {measurement} since {global_since} (global min across locations and fields)")

            # fetch data once per measurement (hydro/smn) with all locations (bern, thun) and fields (tt, ss, rr)
            df = await fetch_influx(store, meas_fields, global_since, mirrored_at)
            if df is None:
                logger.info(f"No new data for {measurement}")
                continue

            # upsert data per location for less impact on the postgres db and more granular time control (prob overkill)
            for location_orig, loc_fields in grp_loc_fields.items():
                since = since_per_loc[location_orig]
                await upsert_measurement_location(df, table_name, location_orig, mirrored_at, loc_fields, since, pool)

    logger.info(f"Mirroring finished in {datetime.now(UTC) - mirrored_at}")


async def create_measurement_table(measurement: str, fields: list[FieldRequest], pool: AsyncConnectionPool) -> str:
    table_name = TABLE_NAME_PREFIX + measurement

    # create tables and or add columns. do not drop columns we don't store anymore, they just stay untouched.
    # in this context, there's only one row per time & loc, like in influx. run_ts is just to know when the data was fetched.
    async with pool.connection() as conn:
        await conn.execute(
            sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        time timestamptz NOT NULL,
                        location text NOT NULL,
                        location_upstream text NOT NULL,
                        mirrored_at timestamptz NOT NULL,
                        PRIMARY KEY (time, location)
                    )
                """).format(table=sql.Identifier(table_name))
        )
        await make_hypertable(conn, table_name)
        for fr in fields:
            await conn.execute(
                sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} real").format(
                    table=sql.Identifier(table_name), col=sql.Identifier(fr.field)
                )
            )

    return table_name


async def get_latest_times(
    meas_table_name: str, loc_fields: dict[str, list[FieldRequest]], pool: AsyncConnectionPool
) -> dict[str, datetime]:
    # only count rows where ALL requested fields for that location are non-null.
    # when a new column is added (all NULLs), max_time returns NULL -> DEFAULT_SINCE.
    # it can even handle a column being removed and re-added later; it will pick back up where it left off.
    loc_clauses: list[sql.Composed] = []
    for location_orig, fields in loc_fields.items():
        not_null_checks = sql.SQL(" AND ").join(
            sql.SQL("{col} IS NOT NULL").format(col=sql.Identifier(fr.field)) for fr in fields
        )
        loc_clauses.append(
            sql.SQL("(location = {loc} AND {checks})").format(
                loc=sql.Literal(location_orig.upper()), checks=not_null_checks
            )
        )

    async with pool.connection() as conn:
        query = sql.SQL("SELECT location, MAX(time) AS max_time FROM {table} WHERE {clauses} GROUP BY location").format(
            table=sql.Identifier(meas_table_name),
            clauses=sql.SQL(" OR ").join(loc_clauses),
        )
        logger.debug(f"Selecting latest times in postgres db: {query.as_string()}")
        df_max = await copy_to_df(conn, query)

    since_per_loc: dict[str, datetime] = {}
    for location_orig in loc_fields.keys():
        rows = df_max[df_max["location"] == location_orig] if not df_max.empty else df_max
        since_per_loc[location_orig] = (
            datetime.fromisoformat(cast(str, rows.iloc[0]["max_time"])) + timedelta(seconds=1)
            if not rows.empty
            else DEFAULT_SINCE
        )

    return since_per_loc


async def fetch_influx(
    store: RemoteExistenzStore, fields: Sequence[FieldRequest], since: datetime, until: datetime
) -> DataFrame | None:
    try:
        # TODO it would be nicer if store.query returned an empty df (or None), but that reaches deep, no time
        df = store.query((since, until), fields)
    except ValueError as e:
        # make sure not to catch any actual errors by accident
        if "no data" not in str(e):
            raise

        return None

    # resample to drop the last timestamp influx returns, no clue why it does that...
    # this is equivalent to aare_train.preparation.resample; didn't want to split it yet, but.. TODO
    df = df.set_index(TIME).resample(EXPECTED_FREQ).first().reset_index(TIME)
    df = df.rename(columns={TIME: "time"})

    return df


async def upsert_measurement_location(
    df: DataFrame,
    table_name: str,
    location_orig: str,
    mirrored_at: datetime,
    fields: list[FieldRequest],
    since: datetime,
    pool: AsyncConnectionPool,
):
    location_up = fields[0].location  # will all have the same
    assert all(fr.location == location_up for fr in fields), "Not all fields have the same upstream location!"

    df_name_map = {fr.name: fr.field for fr in fields}
    # select only fields for this measurement(=table) and location from big df
    loc_df = df[["time"] + list(df_name_map.keys())].copy()
    # rename original field_loc names to just field
    loc_df = loc_df.rename(columns=df_name_map)
    # filter to upsert less data, maybe this measurement@loc doesn't have new data
    loc_df = loc_df[loc_df["time"] > since]

    if loc_df.empty:
        logger.info(f"No new data for {table_name}@{location_orig}")
        return

    loc_df["location"] = location_orig
    loc_df["location_upstream"] = location_up
    loc_df["mirrored_at"] = mirrored_at

    conflict_cols = ["time", "location"]
    # update mirrored_at and all field columns. time & location obv. stay the same (otherwise no conflict would occur).
    # location_upstream is also updated to avoid desync if we ever changed the mapping from our to upstream locations.
    update_cols = ["mirrored_at", "location_upstream"] + list(df_name_map.values())

    async with pool.connection() as conn:
        await upsert_df(conn, loc_df, table_name, conflict_cols, update_cols)

    logger.info(f"Upserted {len(loc_df)} rows into {table_name} for {location_orig}")


if __name__ == "__main__":
    uvloop.run(main())
