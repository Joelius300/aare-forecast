import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from aare_timescale.timescale import make_hypertable
import pandas as pd
from psycopg import AsyncConnection, sql
from psycopg_pool import AsyncConnectionPool

from aare_timescale.postgres import copy_from_df, copy_to_df

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

    async def select(
        self,
        columns: Sequence[str] | None = None,
        *,
        where: sql.SQL | sql.Composed | None = None,
        sort_by: str | None = "time",
        time_cols: Sequence[str] = ("run_ts", "time"),
    ) -> pd.DataFrame:
        """Select from table with very simple options. to be extended if necessary."""

        cols = columns or self.columns
        assert cols is not None
        query = sql.SQL("""select {cols} from {table} {where} {sort}""").format(
            cols=sql.SQL(", ").join([sql.Identifier(c) for c in cols]),
            table=sql.Identifier(self.table_name),
            where="" if not where else sql.SQL("WHERE ") + where,
            sort="" if not sort_by else sql.SQL("ORDER BY {0}").format(sql.Identifier(sort_by)),
        )
        logger.debug(f"Selecting from '{self.table_name}' with query: {query.as_string()}")

        async with self.connection_pool.connection() as conn:
            df = await copy_to_df(conn, query)

        for c in time_cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])

        return df

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

            # note, since copy_from_df also specifies the df columns and their order, it's probably not needed to add
            # these empty columns anymore. With None, it explicitly puts null, if you omit it, it will take the default
            # which could be configured to something else per column, so there is a semantic difference in some cases.

            # add empty columns, for this we need a copy, otherwise the original df would get the additional columns too
            to_store = to_store.copy()
            # to_csv omits None and np.nan the same way so it shouldn't matter. None has dtype object, nan float64.
            to_store[list(missing_cols)] = None

        return to_store[self.columns]  # align column order, should have all required columns now (or have raised)
