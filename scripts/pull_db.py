import argparse
import asyncio
from collections.abc import Sequence
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
import subprocess

from aare_timescale.postgres import copy_from_df, copy_to_df
import psycopg
from psycopg import sql
import uvloop

from aare_train.paths import DATA_FOLDER

logger = logging.getLogger(__name__)

DEFAULT_SINCE = datetime(2026, 1, 1).astimezone()
last_sync_file = DATA_FOLDER / ".last_db_sync"
pg_connect = psycopg.AsyncConnection.connect


def dokku_cmd(host: str, *cmd: str):
    return subprocess.run(["ssh", f"dokku@{host}"] + list(cmd))


class ExposePostgresPort:
    def __init__(self, dokku_host: str, db_name: str, port: int):
        self.dokku_host = dokku_host
        self.db_name = db_name
        self.port = port

    def __enter__(self):
        # dokku postgres:expose {db_name} {port}
        dokku_cmd(self.dokku_host, "postgres:expose", self.db_name, str(self.port))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # dokku postgres:unexpose {db_name}
        dokku_cmd(self.dokku_host, "postgres:unexpose", self.db_name)


@dataclass
class CliArgs:
    db_name: str
    expose_port: int
    source_db_user: str
    source_db_password: str
    destination_db: str
    tables: list[str]
    since: datetime
    clear: bool
    dokku_host: str = ""
    should_update_timestamp: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pull from remote/prod db into your local/staging one")
    parser.add_argument("--db-name", help="Name of remote postgres db on dokku", default="aare-oraku-forecast")
    parser.add_argument("--expose-port", help="Port remote db is exposed on", default=None)
    parser.add_argument(
        "--destination-db",
        help="Connection string for destination db",
        default="host=127.0.0.1 dbname=aare_oraku user=postgres password=password",
    )
    parser.add_argument(
        "--tables", help="Tables to sync", nargs="+", default=("forecast", "forecast_meta", "meteotest", "bafu_flow")
    )
    parser.add_argument(
        "--since",
        help="How far back to pull data into your local db. Will use and update .last_db_sync if not provided.",
        default=None,
        type=datetime.fromisoformat,
    )
    parser.add_argument(
        "--clear",
        help="If a delete statement with 'where run_ts>since' should be run before insertion.",
        default=False,
        action="store_true",
    )
    parser.add_argument("--source-db-user", help="User for the source db", default="reader")
    parser.add_argument("--source-db-password", help="Password for the source db user", default="mängischlängtläse")

    args_namespace: argparse.Namespace = parser.parse_args()
    args = CliArgs(**vars(args_namespace))  # pyright: ignore[reportAny]

    host = os.environ.get("DOKKU_HOST")
    if not host:
        raise ValueError("Must set DOKKU_HOST")
    args.dokku_host = host

    if not args.expose_port:
        args.expose_port = random.randint(2000, 48000)

    args.should_update_timestamp = False
    if not args.since:
        args.should_update_timestamp = True
        args.since = datetime.fromisoformat(last_sync_file.read_text()) if last_sync_file.exists() else DEFAULT_SINCE

    return args


async def main():
    logging.basicConfig(level="DEBUG")
    args = parse_args()
    if args.clear:
        logger.warning(
            f"Are you sure that you want to delete all data where run_ts >= {args.since.isoformat()} "
            + f"in the tables {', '.join(args.tables)} of database '{args.destination_db}'?"
        )
        if input("Sure? (y/N)").lower() != "y":
            print("cancelling")
            return

    with ExposePostgresPort(args.dokku_host, args.db_name, args.expose_port):
        async with await pg_connect(
            f"host={args.dokku_host} port={args.expose_port} dbname={args.db_name.replace('-', '_')} "
            + f"user={args.source_db_user} password={args.source_db_password}"
        ) as source_db:
            async with await pg_connect(args.destination_db) as dest_db:
                now = datetime.now(UTC)
                await copy_data(source_db, dest_db, args.tables, args.since, args.clear)
                if args.should_update_timestamp:
                    last_sync_file.write_text(now.isoformat())


async def copy_data(
    source_db: psycopg.AsyncConnection,
    dest_db: psycopg.AsyncConnection,
    tables: Sequence[str],
    since: datetime,
    clear: bool,
):
    await asyncio.gather(*(copy_table(source_db, dest_db, table, since, clear) for table in tables))


async def copy_table(
    source_db: psycopg.AsyncConnection, dest_db: psycopg.AsyncConnection, table: str, since: datetime, clear: bool
):
    logger.debug("Start copying table '{table}'")
    if clear:
        logger.debug(f"Clearing table 'table' after {since}")
        await dest_db.execute(
            "DELETE FROM {table} WHERE run_ts >= {since}", dict(table=sql.Identifier(table), since=since)
        )

    # sorting by first and second column -> run_ts & time or run_ts and whatever. just so it's prettier.
    data = await copy_to_df(
        source_db,
        sql.SQL("""
        SELECT * FROM {table}
        WHERE run_ts >= {since}
        ORDER BY 1, 2
        """).format(table=sql.Identifier(table), since=since),
    )
    logger.debug(f"Pulled {len(data)} rows with {len(data.columns)} columns from the source db (table '{table}').")
    # could also use binary COPY TO STDOUT, loop through the chunks and write them to the binary COPY FROM STDIN
    # for better performance esp. if there is a lot of data. this is just super simple, debuggable and lenient.
    # TODO I didn't think it mattered but it took 5min to sync 4 months worth of data, that's really bad.
    # https://www.psycopg.org/psycopg3/docs/basic/copy.html#example-copying-a-table-across-servers
    await copy_from_df(dest_db, data, table)
    logger.debug(f"Inserted {len(data)} rows into '{table}' on destination db.")


if __name__ == "__main__":
    uvloop.run(main())
