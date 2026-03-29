from datetime import UTC, datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import logging

import configargparse
import uvloop
import pandas as pd
from psycopg import sql

from aare_logging.logging import setup_logging
from aare_timescale.postgres import init_db_pool, copy_to_df, copy_from_df
from aare_timescale.timescale import make_hypertable
from aare_influx.remote_existenz_store import RemoteExistenzStore
from aare_influx.field_request import FieldRequest

logger = logging.getLogger(__name__)

DEFAULT_SINCE = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass
class CliArgs:
    connection_string: str
    fields: list[str]
    logging_level: str
    loki_url: str | None
    loki_password: str | None


def parse_cli_args() -> CliArgs:
    default_config_file = Path(__file__).parent / "dev_config.yaml"
    p = configargparse.ArgParser(auto_env_var_prefix="oraku_", default_config_files=[str(default_config_file)])
    p.add_argument("-c", "--connection-string", required=True, type=str, help="Connection string for the timescale db")
    p.add_argument(
        "-f",
        "--fields",
        required=True,
        nargs="+",
        type=str,
        help='FieldRequest strings, e.g. "hydro/temperature:mean_1h@bern"',
    )
    p.add_argument("--logging-level", default="INFO", type=str)
    p.add_argument("--loki-url", default=None, type=str)
    p.add_argument("--loki-password", default=None, type=str)
    return CliArgs(**vars(p.parse_args()))  # pyright: ignore[reportAny]


async def main():
    args = parse_cli_args()
    setup_logging(args.logging_level, args.loki_url, args.loki_password, "aare-oraku-influx2pg")

    run_ts = datetime.now(UTC)
    requests = [FieldRequest.from_str(f) for f in args.fields]

    # Group by (measurement, location_orig) — one influx query per group
    groups: dict[tuple[str, str], list[FieldRequest]] = defaultdict(list)
    for fr in requests:
        groups[(fr.measurement, str(fr.location_orig))].append(fr)

    store = RemoteExistenzStore()

    async with init_db_pool(args.connection_string) as pool:
        # Ensure all tables and columns exist, grouped by measurement
        measurements_fields: dict[str, list[FieldRequest]] = defaultdict(list)
        for fr in requests:
            measurements_fields[fr.measurement].append(fr)

        for measurement, meas_fields in measurements_fields.items():
            table_name = f"mirror_{measurement}"
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

        # Mirror each (measurement, location) group
        for (measurement, location_orig), fields in groups.items():
            table_name = f"mirror_{measurement}"
            location = location_orig.upper()

            # Query latest mirrored timestamp for this location
            async with pool.connection() as conn:
                df_max = await copy_to_df(
                    conn,
                    sql.SQL("SELECT MAX(time) AS max_time FROM {table} WHERE location = {loc}").format(
                        table=sql.Identifier(table_name), loc=sql.Literal(location)
                    ),
                )

            max_time = df_max.iloc[0]["max_time"]
            if pd.isna(max_time):
                since = DEFAULT_SINCE
            else:
                since = pd.Timestamp(max_time).to_pydatetime() + timedelta(seconds=1)

            logger.info(f"Fetching {measurement}/{location} since {since}")

            # Fetch from influx (synchronous)
            df = store.query((since, run_ts), fields)

            if df.empty:
                logger.info(f"No new data for {measurement}/{location}")
                continue

            # Transform: rename columns, add location and run_ts
            df = df.rename(columns={"_time": "time"})
            df = df.rename(columns={fr.name: fr.field for fr in fields})
            df["location"] = location
            df["run_ts"] = run_ts

            async with pool.connection() as conn:
                await copy_from_df(conn, df, table_name)

            logger.info(f"Inserted {len(df)} rows into {table_name} for {location}")

    logger.info(f"Mirror run finished in {datetime.now(UTC) - run_ts}")


if __name__ == "__main__":
    uvloop.run(main())
