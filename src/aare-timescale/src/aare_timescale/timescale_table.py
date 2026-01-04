import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from aare_timescale.timescale import make_hypertable
import pandas as pd
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from aare_timescale.postgres import copy_from_df

logger = logging.getLogger(__name__)


# micro ORM for our use-case :)
# currently no abstraction because no intention of switching database. would need a big refactor anyway.
class TimescaleTable(ABC):
    def __init__(
        self,
        connection_pool: AsyncConnectionPool,
        table_name: str,
        columns: Sequence[str],
        allow_extra_columns: bool = False,
        allow_missing_columns: bool = False,
    ):
        self.connection_pool: AsyncConnectionPool = connection_pool
        self.table_name: str = table_name
        self.columns: Sequence[str] = columns
        self._allow_extra_columns: bool = allow_extra_columns
        self._allow_missing_columns: bool = allow_missing_columns

    @abstractmethod
    async def ensure_table_exists(self):
        """
        Create the table with if necessary. Order of the columns must match the columns parameter.
        THIS FUNCTION MUST BE IDEMPOTENT (can be executed multiple times without side effects).
        """
        # If a new column is added, it should be done with something like 'alter add if not exists'.
        pass

    async def insert(self, df: pd.DataFrame):
        to_store = self._align_columns(df)
        async with self.connection_pool.connection() as conn:
            await copy_from_df(conn, to_store, self.table_name)

    async def make_hypertable(self, conn: AsyncConnection, time_col: str = "time", chunk_interval: int = 7):
        """Make table into a hypertable if necessary."""
        await make_hypertable(conn, self.table_name, time_col, chunk_interval)

    def _align_columns(self, df: pd.DataFrame):
        expected_cols = set(self.columns)
        actual_cols = set(df.columns)
        extra_cols = actual_cols - expected_cols
        missing_cols = expected_cols - actual_cols

        to_store = df
        if len(extra_cols) > 0:
            extra_cols_msg = "Column mismatch; Got extra columns: " + ", ".join(extra_cols)
            if not self._allow_extra_columns:
                raise ValueError(extra_cols_msg)

            logger.warning(extra_cols_msg)
            # select only the columns we care about (without failing if some are missing). no need to copy here.
            # this is just a small optimization to avoid copying more than we need if we have missing cols.
            to_store = to_store[list(expected_cols & actual_cols)]

        if len(missing_cols) > 0:
            missing_cols_msg = "Column mismatch; Missing columns: " + ", ".join(missing_cols)
            if not self._allow_missing_columns:
                raise ValueError(missing_cols_msg)

            logger.warning(missing_cols_msg)
            # add empty columns, for this we need a copy, otherwise the original df would get the additional columns too
            to_store = to_store.copy()
            # to_csv omits None and np.nan the same way so it shouldn't matter. None has dtype object, nan float64.
            to_store[list(missing_cols)] = None

        return to_store[self.columns]  # align column order, should have all required columns now (or have raised)
