from datetime import UTC, datetime, timedelta
from collections import defaultdict
import logging

import psycopg
import uvloop
import pandas as pd
from psycopg import sql

from aare_logging.logging import setup_logging
from aare_timescale.postgres import init_db_pool, copy_to_df, copy_from_df
from aare_timescale.timescale import make_hypertable
from aare_influx.remote_existenz_store import RemoteExistenzStore
from aare_influx.field_request import FieldRequest
from oraku_influx2pg.args import parse_cli_args

DEFAULT_SINCE = datetime(2026, 1, 1, tzinfo=UTC)

logger = logging.getLogger(__name__)

TABLE_NAME_PREFIX = "mirror_"


async def _upsert_df(conn: psycopg.AsyncConnection, df: pd.DataFrame, table_name: str, field_names: list[str]):
    """Copy df into a temp table then upsert into the target on (time, location) conflict."""
    temp_name = f"_tmp_{table_name}"
    await conn.execute(
        sql.SQL("CREATE TEMP TABLE {tmp} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP").format(
            tmp=sql.Identifier(temp_name), table=sql.Identifier(table_name)
        )
    )
    await copy_from_df(conn, df, temp_name)
    update_cols = [sql.Identifier(c) for c in ["run_ts"] + field_names]
    update_set = sql.SQL(", ").join(sql.SQL("{col} = EXCLUDED.{col}").format(col=col) for col in update_cols)
    await conn.execute(
        sql.SQL("INSERT INTO {table} SELECT * FROM {tmp} ON CONFLICT (time, location) DO UPDATE SET {updates}").format(
            table=sql.Identifier(table_name),
            tmp=sql.Identifier(temp_name),
            updates=update_set,
        )
    )


async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-influx2pg")

    run_ts = datetime.now(UTC)
    requests = [FieldRequest.from_str(f) for f in args.fields]
    store = RemoteExistenzStore()

    async with init_db_pool(args.connection_string) as pool:
        measurements_fields: dict[str, list[FieldRequest]] = defaultdict(list)
        for fr in requests:
            measurements_fields[fr.measurement].append(fr)

        for measurement, meas_fields in measurements_fields.items():
            table_name = TABLE_NAME_PREFIX + measurement

            # Ensure table, hypertable, columns, and unique constraint exist
            async with pool.connection() as conn:
                await conn.execute(
                    sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        time timestamptz NOT NULL,
                        run_ts timestamptz NOT NULL,
                        location text NOT NULL,
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

            # Group fields by location (using location_orig as key)
            loc_fields: dict[str, list[FieldRequest]] = defaultdict(list)
            for fr in meas_fields:
                loc_fields[str(fr.location_orig)].append(fr)

            # Fetch latest mirrored timestamp per location in a single query.
            # Only count rows where ALL requested fields for that location are non-null:
            # when a new column is added (all NULLs), max_time returns NULL → DEFAULT_SINCE.
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
            for location_orig, _ in loc_fields.items():
                location = location_orig.upper()
                rows = df_max[df_max["location"] == location] if not df_max.empty else pd.DataFrame()
                if rows.empty or pd.isna(rows.iloc[0]["max_time"]):
                    since_per_loc[location] = DEFAULT_SINCE
                else:
                    since_per_loc[location] = pd.Timestamp(rows.iloc[0]["max_time"]).to_pydatetime() + timedelta(
                        seconds=1
                    )

            global_since = min(since_per_loc.values())
            logger.info(f"Fetching {measurement} since {global_since} (global min across locations)")

            df = store.query((global_since, run_ts), meas_fields)

            if df.empty:
                logger.info(f"No new data for {measurement}")
                continue

            df = df.rename(columns={"_time": "time"})

            for location_orig, fields in loc_fields.items():
                location = location_orig.upper()
                since = since_per_loc[location]

                col_names = [fr.name for fr in fields]
                loc_df = df[["time"] + col_names].copy()
                loc_df = loc_df.rename(columns={fr.name: fr.field for fr in fields})
                loc_df = loc_df[loc_df["time"] > since]

                if loc_df.empty:
                    logger.info(f"No new data for {measurement}/{location}")
                    continue

                loc_df["location"] = location
                loc_df["run_ts"] = run_ts
                field_names = [fr.field for fr in fields]
                loc_df = loc_df[["time", "run_ts", "location"] + field_names]

                async with pool.connection() as conn:
                    await _upsert_df(conn, loc_df, table_name, field_names)

                logger.info(f"Upserted {len(loc_df)} rows into {table_name} for {location}")

    logger.info(f"Mirror run finished in {datetime.now(UTC) - run_ts}")


if __name__ == "__main__":
    uvloop.run(main())
