from datetime import UTC, datetime, timedelta
from collections import defaultdict
import logging
from typing import cast

import psycopg
import uvloop
import pandas as pd
from psycopg import sql

from aare.constants import TIME
from aare_logging.logging import setup_logging
from aare_timescale.postgres import init_db_pool, copy_to_df, copy_from_df
from aare_timescale.timescale import make_hypertable
from aare_influx.remote_existenz_store import RemoteExistenzStore
from aare_influx.field_request import FieldRequest
from oraku_influx2pg.args import parse_cli_args

DEFAULT_SINCE = datetime(2026, 1, 1, tzinfo=UTC)

logger = logging.getLogger(__name__)

TABLE_NAME_PREFIX = "mirror_"
EXPECTED_FREQ = "1h"  # not supporting anything else right now


async def _upsert_df(conn: psycopg.AsyncConnection, df: pd.DataFrame, table_name: str, val_cols: list[str]):
    """Copy df into a temp table then upsert into the target on (time, location) conflict (also updating run_ts)."""
    # this fancy method is something claude came up with, and I'm glad it did. I'm not that deep in the postgres game.
    temp_name = f"_tmp_{table_name}"
    await conn.execute(
        sql.SQL("CREATE TEMP TABLE {tmp} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP").format(
            tmp=sql.Identifier(temp_name), table=sql.Identifier(table_name)
        )
    )

    await copy_from_df(conn, df, temp_name)
    # update run_ts and all provided value columns. time, location must stay the same (otherwise no conflict would occur).
    # location_upstream is also updated to avoid desync if we ever changed the mapping from our to upstream locations.
    update_cols = [sql.Identifier(c) for c in ["run_ts", "location_upstream"] + val_cols]
    # "EXCLUDED." references the row that would have been inserted but led to the conflict.
    # just "col" or "table.col" would be the value that is already there in the existing row.
    update_set = sql.SQL(", ").join(sql.SQL("{col} = EXCLUDED.{col}").format(col=col) for col in update_cols)
    await conn.execute(
        sql.SQL("INSERT INTO {table} SELECT * FROM {tmp} ON CONFLICT (time, location) DO UPDATE SET {updates}").format(
            table=sql.Identifier(table_name),
            tmp=sql.Identifier(temp_name),
            updates=update_set,
        )
    )


# TODO refactor into a few methods for table creation, latest fetching, fetching data, and insertion
async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-influx2pg")

    run_ts = datetime.now(UTC)
    requests = [FieldRequest.from_str(f) for f in args.fields]
    store = RemoteExistenzStore()

    async with init_db_pool(args.connection_string) as pool:
        measurements_fields: dict[str, list[FieldRequest]] = defaultdict(list)
        for fr in requests:
            assert fr.freq == EXPECTED_FREQ, f"Got unexpected frequency '{fr.freq}' (instead of '{EXPECTED_FREQ}')"
            measurements_fields[fr.measurement].append(fr)

        for measurement, meas_fields in measurements_fields.items():
            table_name = TABLE_NAME_PREFIX + measurement

            # create tables and or add columns. do not drop columns we don't store anymore, they just stay untouched.
            # in this context, there's only one row per time, like in influx. run_ts is just to know when the data was fetched.
            async with pool.connection() as conn:
                await conn.execute(
                    sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        time timestamptz NOT NULL,
                        run_ts timestamptz NOT NULL,
                        location text NOT NULL,
                        location_upstream text NOT NULL,
                        PRIMARY KEY (time, location)
                    )
                """).format(table=sql.Identifier(table_name))
                )
                await make_hypertable(conn, table_name)
                for fr in meas_fields:
                    await conn.execute(
                        sql.SQL("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} real").format(
                            table=sql.Identifier(table_name), col=sql.Identifier(fr.field)
                        )
                    )

            # group fields by location[_orig] since all variables share a row per time and location
            loc_fields: dict[str, list[FieldRequest]] = defaultdict(list)
            for fr in meas_fields:
                loc_fields[str(fr.location_orig).upper()].append(fr)

            # fetch latest mirrored timestamp per location.
            # only count rows where ALL requested fields for that location are non-null.
            # when a new column is added (all NULLs), max_time returns NULL → DEFAULT_SINCE.
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
                query = sql.SQL(
                    "SELECT location, MAX(time) AS max_time FROM {table} WHERE {clauses} GROUP BY location"
                ).format(
                    table=sql.Identifier(table_name),
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

            global_since = min(since_per_loc.values())
            logger.info(f"Fetching {measurement} since {global_since} (global min across locations and columns)")

            try:
                # TODO it would be nicer if store.query returned an empty df (or None), but that reaches deep, no time
                df = store.query((global_since, run_ts), meas_fields)
            except ValueError as e:
                # make sure not to catch any actual errors by accident
                if "no data" not in str(e):
                    raise

                logger.info(f"No new data for {measurement}")
                continue

            # resample to drop the last timestamp influx returns, no clue why it does that...
            # this is equivalent to aare_train.preparation.resample; didn't want to split it yet, but.. TODO
            df = df.set_index(TIME).resample(EXPECTED_FREQ).first().reset_index(TIME)
            df = df.rename(columns={TIME: "time"})

            for location_orig, fields in loc_fields.items():
                location_up = fields[0].location  # will all have the same
                since = since_per_loc[location_orig]

                col_names = [fr.name for fr in fields]
                loc_df = df[["time"] + col_names].copy()
                loc_df = loc_df.rename(columns={fr.name: fr.field for fr in fields})
                loc_df = loc_df[loc_df["time"] > since]  # filter to upsert less data

                if loc_df.empty:
                    logger.info(f"No new data for {measurement}@{location_orig}")
                    continue

                loc_df["location"] = location_orig
                loc_df["location_upstream"] = location_up
                loc_df["run_ts"] = run_ts
                field_names = [fr.field for fr in fields]
                async with pool.connection() as conn:
                    await _upsert_df(conn, loc_df, table_name, field_names)

                logger.info(f"Upserted {len(loc_df)} rows into {table_name} for {location_orig}")

    logger.info(f"Mirror run finished in {datetime.now(UTC) - run_ts}")


if __name__ == "__main__":
    uvloop.run(main())
