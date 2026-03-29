from datetime import UTC, datetime, timedelta
from collections import defaultdict
import logging

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


async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-influx2pg")

    run_ts = datetime.now(UTC)
    requests = [FieldRequest.from_str(f) for f in args.fields]
    store = RemoteExistenzStore()

    async with init_db_pool(args.connection_string) as pool:
        # Group requests by measurement (= one postgres table each)
        measurements_fields: dict[str, list[FieldRequest]] = defaultdict(list)
        for fr in requests:
            measurements_fields[fr.measurement].append(fr)

        for measurement, meas_fields in measurements_fields.items():
            table_name = TABLE_NAME_PREFIX + measurement

            # Ensure table and all columns exist
            async with pool.connection() as conn:
                await conn.execute(
                    sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        time timestamptz NOT NULL,
                        run_ts timestamptz NOT NULL,
                        location text NOT NULL
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

            # Collect unique locations (uppercase, as stored in postgres)
            locations_upper = list({str(fr.location_orig).upper() for fr in meas_fields})

            # Fetch latest mirrored timestamp per location in a single query
            async with pool.connection() as conn:
                df_max = await copy_to_df(
                    conn,
                    sql.SQL(
                        "SELECT location, MAX(time) AS max_time FROM {table}"
                        " WHERE location = ANY({locs}) GROUP BY location"
                    ).format(
                        table=sql.Identifier(table_name),
                        locs=sql.Literal(locations_upper),
                    ),
                )

            since_per_loc: dict[str, datetime] = {}
            for loc in locations_upper:
                rows = df_max[df_max["location"] == loc] if not df_max.empty else pd.DataFrame()
                if rows.empty or pd.isna(rows.iloc[0]["max_time"]):
                    since_per_loc[loc] = DEFAULT_SINCE
                else:
                    since_per_loc[loc] = pd.Timestamp(rows.iloc[0]["max_time"]).to_pydatetime() + timedelta(seconds=1)

            global_since = min(since_per_loc.values())
            logger.info(f"Fetching {measurement} since {global_since} (global min across locations)")

            # One influx call for all fields of this measurement
            df = store.query((global_since, run_ts), meas_fields)

            if df.empty:
                logger.info(f"No new data for {measurement}")
                continue

            df = df.rename(columns={"_time": "time"})

            # Group fields by location so we can split the wide df per location
            loc_fields: dict[str, list[FieldRequest]] = defaultdict(list)
            for fr in meas_fields:
                loc_fields[str(fr.location_orig)].append(fr)

            for location_orig, fields in loc_fields.items():
                location = location_orig.upper()
                since = since_per_loc[location]

                col_names = [fr.name for fr in fields]
                loc_df = df[["time"] + col_names].copy()
                loc_df = loc_df.rename(columns={fr.name: fr.field for fr in fields})

                # Drop rows already present in postgres for this location
                loc_df = loc_df[loc_df["time"] > since]

                if loc_df.empty:
                    logger.info(f"No new data for {measurement}/{location}")
                    continue

                loc_df["location"] = location
                loc_df["run_ts"] = run_ts
                loc_df = loc_df[["time", "run_ts", "location"] + [fr.field for fr in fields]]

                async with pool.connection() as conn:
                    await copy_from_df(conn, loc_df, table_name)

                logger.info(f"Inserted {len(loc_df)} rows into {table_name} for {location}")

    logger.info(f"Mirror run finished in {datetime.now(UTC) - run_ts}")


if __name__ == "__main__":
    uvloop.run(main())
