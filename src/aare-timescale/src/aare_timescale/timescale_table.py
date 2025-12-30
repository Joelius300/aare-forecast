import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

from aare_timescale.timescale import make_hypertable
import pandas as pd
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from aare_timescale.postgres import copy_from_df

logger = logging.getLogger(__name__)


# micro ORM for our use-case :)
# currently no abstraction because no intention of switching database. would need a big refactor anyway.
class TimescaleTable(ABC):
    def __init__(self, connection_pool: AsyncConnectionPool, table_name: str, columns: Sequence[str]):
        self.connection_pool: AsyncConnectionPool = connection_pool
        self.table_name: str = table_name
        self.columns: Sequence[str] = columns

    @abstractmethod
    async def ensure_table_exists(self):
        """
        Create the table with if necessary. Order of the columns must match the columns parameter.
        THIS FUNCTION SHOULD BE IDEMPOTENT (can be executed multiple times without side effects).
        """
        # If a new column is added, it should be done with something like 'alter add if not exists'.
        pass

    async def insert(self, df: pd.DataFrame):
        to_store = cast(pd.DataFrame, df[self.columns])  # if not all columns are in df, it will fail here already
        # TODO maybe allow for less columns, fill them with null. Pulling data from external services should be lenient,
        #  the forecast service can still say it doesn't have everything it needs to make a forecast.
        if len(self.columns) != len(df.columns):
            # more columns than required
            logger.warning(
                f"Column mismatch of dataframe and table. Expected: {','.join(self.columns)} but got {','.join(df.columns)}"
            )

        async with self.connection_pool.connection() as conn:
            await copy_from_df(conn, to_store, self.table_name)

    async def make_hypertable(self, conn: AsyncConnection, time_col: str = "time", chunk_interval: int = 7):
        """Make table into a hypertable if necessary."""
        await make_hypertable(conn, self.table_name, time_col, chunk_interval)
