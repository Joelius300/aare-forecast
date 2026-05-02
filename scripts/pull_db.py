import argparse
import asyncio
from collections.abc import Sequence
import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
import subprocess

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
    verbose: bool
    dokku_host: str = ""
    should_update_timestamp: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description="Pull from remote/prod db into your local/staging one")
    parser.add_argument("--db-name", help="Name of remote postgres db on dokku", default="aare-oraku-forecast")
    parser.add_argument("--expose-port", help="Port remote db is exposed on", default=None)
    parser.add_argument("--source-db-user", help="User for the source db", default="reader")
    parser.add_argument("--source-db-password", help="Password for the source db user", default="mängischlängtläse")
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
    parser.add_argument(
        "--verbose",
        help="More logs",
        default=False,
        action="store_true",
    )

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
        # read from .last_db_sync if not explicitly specified. update the file after completion.
        args.should_update_timestamp = True
        args.since = datetime.fromisoformat(last_sync_file.read_text()) if last_sync_file.exists() else DEFAULT_SINCE
    else:
        # interpret as local time if not specified
        args.since = args.since if args.since.tzinfo is not None else args.since.astimezone()

    return args


async def main():
    args = parse_args()
    logging.basicConfig(level="INFO" if not args.verbose else "DEBUG")
    if args.clear:
        logger.warning(
            f"Are you sure that you want first to delete all data where run_ts >= {args.since.isoformat()} "
            + f"in the tables {', '.join(args.tables)} of database '{args.destination_db}'?"
        )

        if input("Sure? (y/N)").lower() != "y":
            logger.info("aborting")
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
    logger.debug(f"Start syncing table '{table}'")
    table_since_params = dict(table=sql.Identifier(table), since=since)
    if clear:
        logger.info(f"Clearing table '{table}' after {since}")
        await dest_db.execute(sql.SQL("DELETE FROM {table} WHERE run_ts >= {since}").format(**table_since_params))

    count = await (
        await source_db.execute(
            sql.SQL("SELECT count(*) FROM {table} WHERE run_ts >= {since}").format(**table_since_params)
        )
    ).fetchone()
    count = count[0] if count else None
    assert isinstance(count, int) and count >= 0, f"got invalid count: {count}"
    if count == 0:
        logger.info(f"No new data in table '{table}, skipping")
        return

    logger.info(f"Pulling {count} rows from '{table}' on source db.")

    # sorting by first and second column -> run_ts & time or run_ts and whatever. just so it's prettier.
    async with source_db.cursor().copy(
        sql.SQL("""COPY (
            SELECT * FROM {table}
            WHERE run_ts >= {since}
            ORDER BY 1, 2
        ) TO STDOUT (FORMAT BINARY)
        """).format(**table_since_params),
    ) as source_copy:
        async with dest_db.cursor().copy(
            sql.SQL("COPY {table} FROM STDIN (FORMAT BINARY)").format(table=sql.Identifier(table))
        ) as dest_copy:
            async for chunk in source_copy:
                # for this to work, the tables must be exactly identical (down to order and type).
                # also the same postgres and timescale version if possible.
                await dest_copy.write(chunk)

    logger.info(f"Inserted {count} rows into '{table}' on destination db.")


if __name__ == "__main__":
    uvloop.run(main())
