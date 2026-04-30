import argparse
import os
from dataclasses import dataclass
from datetime import datetime
import subprocess

import uvloop


def dokku_cmd(host: str, *cmd: str):
    return subprocess.run(["ssh", f"dokku@{host}"] + list(cmd))


class ExposePort:
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
    dokku_host: str
    db_name: str
    expose_port: int
    destination_db: str
    tables: list[str]
    since: datetime
    should_update_timestamp: bool
    clear: bool


def parse_args():
    host = os.environ.get("DOKKU_HOST")
    if not host:
        raise ValueError("Must set DOKKU_HOST")

    parser = argparse.ArgumentParser(description="Pull from remote/prod db into your local/staging one")
    parser.add_argument("--db-name", help="Name of remote postgres db on dokku", default="aare-oraku-forecast")
    parser.add_argument("--expose-port", help="Port remote db is exposed on", default=12324)
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
        type=bool,
    )

    # todo you need connection string for source db. make sure to use readonly user. write scripts in some folder,
    #  all manual for now. continuous aggregates will/should also go there for now.
    # todo since = read from .last_db_sync if not provided


async def main():
    pass


if __name__ == "__main__":
    uvloop.run(main())
