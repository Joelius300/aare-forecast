import logging
from io import BytesIO
from typing import cast, LiteralString, Optional

import pandas as pd
import psycopg
from pandas._typing import WriteBuffer
from psycopg import sql
from psycopg.abc import Params
from psycopg_pool import ConnectionPool
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


# micro ORM for our use-case :)
# currently no abstraction because no intention of switching database. would need a big refactor anyway.
class TimescaleTable(ABC):
    def __init__(self, connection_pool: ConnectionPool, table_name: str, columns: list[str]):
        self.connection_pool = connection_pool
        self.table_name = table_name
        self.columns = columns

    @abstractmethod
    def ensure_table_exists(self):
        """
        Create the table with if necessary. Order of the columns must match the columns parameter.
        THIS FUNCTION SHOULD BE IDEMPOTENT (can be executed multiple times without side effects).
        """
        # If a new column is added, it should be done with something like 'alter add if not exists'.
        pass

    def insert(self, df: pd.DataFrame):
        to_store = cast(pd.DataFrame, df[self.columns])  # if not all columns are in df, it will fail here already
        if len(self.columns) != len(df.columns):
            # more columns than required
            logger.warning(
                f"Column mismatch of dataframe and table. Expected: {','.join(self.columns)} but got {','.join(df.columns)}"
            )

        with self.connection_pool.connection() as conn:
            self.copy_from_df(conn, to_store, self.table_name)

    def make_hypertable(self, conn: psycopg.Connection, time_col="time", chunk_interval=7):
        """Call create_hypertable to turn the table into a hypertable. Idempotent."""
        conn.execute(
            # could add second partitioning dimension with add_dimension after create_hypertable
            sql.SQL("""
            SELECT *
            FROM create_hypertable({table}, by_range({time_col}, INTERVAL '{chunk_interval} days'), if_not_exists => TRUE);
            """).format(table=self.table_name, time_col=time_col, chunk_interval=chunk_interval)
            # strings will be wrapped in ''
        )

    @staticmethod
    def copy_from_df(conn: psycopg.Connection, df: pd.DataFrame, table: str):
        """Copy a pandas DataFrame into a Timescale table."""
        # save dataframe to an in-memory buffer
        buffer = BytesIO()
        df.to_csv(cast(WriteBuffer[bytes], buffer), index=False, header=True)

        buffer.seek(0)
        csv = buffer.getvalue()

        with conn.cursor() as cur:
            # for whatever ungodly reason, it will raise 'invalid input syntax for type timestamp with time zone'
            # if you omit the HEADERs, even though they are supposedly ignored completely, but idk man...
            with cur.copy(
                sql.SQL("COPY {table} FROM STDIN WITH CSV HEADER").format(table=sql.Identifier(table))
            ) as copy:
                copy.write(csv)

    @staticmethod
    def copy_to_df(conn: psycopg.Connection, query: sql.SQL | LiteralString, params: Optional[Params] = None):
        """Copy a Timescale query into a pandas DataFrame."""
        if isinstance(query, str):
            query = sql.SQL(query)
        with conn.cursor() as cur:
            with BytesIO() as bio:
                # make sure the query doesn't end with ; somehow
                with cur.copy(sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query), params) as copy:
                    for data in copy:
                        bio.write(data)
                bio.seek(0)

                return pd.read_csv(bio)
