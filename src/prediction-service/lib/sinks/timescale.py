from io import BytesIO
from typing import cast

import pandas as pd
import psycopg
from pandas._typing import WriteBuffer
from psycopg import sql
from psycopg_pool import ConnectionPool

from lib.sinks.sink import Sink


class TimescaleSink(Sink):
    def __init__(self, connection_pool: ConnectionPool):
        self.connection_pool = connection_pool

    def persist(self, table: str, data: pd.DataFrame) -> None:
        with self.connection_pool.connection() as conn:
            self.copy_from_df(conn, data, table)

    @staticmethod
    def copy_from_df(conn: psycopg.Connection, df: pd.DataFrame, table: str):
        """Copy a pandas DataFrame into a Timescale table."""
        # save dataframe to an in-memory buffer
        buffer = BytesIO()
        df.to_csv(cast(WriteBuffer[bytes], buffer), index=False, header=False)

        buffer.seek(0)
        csv = buffer.getvalue()

        with conn.cursor() as cur:
            with cur.copy(
                sql.SQL("COPY {table} FROM STDIN").format(table=sql.Identifier(table))) as copy:
                copy.write(csv)

    @staticmethod
    def copy_to_df(conn: psycopg.Connection, query: sql.SQL):
        """Copy a Timescale query into a pandas DataFrame."""
        with conn.cursor() as cur:
            with BytesIO() as bio:
                # make sure the query doesn't end with ; somehow
                with cur.copy(sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query)) as copy:
                    for data in copy:
                        bio.write(data)
                bio.seek(0)

                return pd.read_csv(bio)
