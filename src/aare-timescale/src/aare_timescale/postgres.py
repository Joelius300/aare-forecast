from io import BytesIO
from typing import TYPE_CHECKING, LiteralString, cast, Any
from collections.abc import Sequence, Mapping

import pandas as pd
import psycopg
from pandas._typing import WriteBuffer
from psycopg import AsyncConnection, sql
from psycopg.abc import Params
from psycopg.rows import TupleRow
from psycopg.sql import Composed, SQL
from psycopg_pool import AsyncConnectionPool

if TYPE_CHECKING:
    import polars as pl

async def copy_to_df(
    conn: AsyncConnection, query: LiteralString | sql.SQL | sql.Composed, params: Params | None = None
) -> pd.DataFrame:
    """Copy a postgres query into a pandas DataFrame."""
    with BytesIO() as bio:
        await _copy_to_csv(conn, params, query, bio)
        bio.seek(0)

        return pd.read_csv(bio)


async def copy_to_df_pl(
        conn: AsyncConnection, query: LiteralString | sql.SQL | sql.Composed, params: Params | None = None
) -> "pl.DataFrame":
    """Copy a postgres query into a polars DataFrame."""
    import polars as pl
    with BytesIO() as bio:
        await _copy_to_csv(conn, params, query, bio)
        bio.seek(0)

        return pl.read_csv(bio)


async def _copy_to_csv(conn: AsyncConnection, params: Sequence[Any] | Mapping[str, Any] | None,
                       query: LiteralString | SQL | Composed, bio: BytesIO) -> None:
    if isinstance(query, str):
        query = sql.SQL(query)
    async with conn.cursor() as cur:
        # make sure the query doesn't end with ; somehow, maybe it's automatic?
        async with cur.copy(
                sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query), params
        ) as copy:
            async for data in copy:
                bio.write(data)


async def copy_from_df(conn: AsyncConnection, df: pd.DataFrame, table: str):
    """
    Copy a pandas DataFrame into a Timescale table.

    Only the columns present in the DataFrame are written; any table columns not
    in the DataFrame receive their default value (typically NULL).
    """
    # save dataframe to an in-memory buffer
    buffer = BytesIO()
    df.to_csv(cast(WriteBuffer[bytes], buffer), index=False, header=True)  # pyright: ignore[reportInvalidCast]

    buffer.seek(0)
    csv = buffer.getvalue()

    col_list = sql.SQL(", ").join(sql.Identifier(c) for c in df.columns)

    async with conn.cursor() as cur:
        # for whatever ungodly reason, it will raise 'invalid input syntax for type timestamp with time zone'
        # if you omit the HEADERs, even though they are supposedly ignored completely, but idk man...
        async with cur.copy(
            sql.SQL("COPY {table} ({cols}) FROM STDIN WITH CSV HEADER").format(
                table=sql.Identifier(table), cols=col_list
            )
        ) as copy:
            await copy.write(csv)


def init_db_pool(
    connection_string: str, min_size: int = 1, max_size: int = 4, prepare_threshold: int | None = 3
) -> AsyncConnectionPool:
    """Initialize a postgres connection pool with some defaults for our small use cases."""
    # Could also use AsyncNullConnectionPool because we probably don't really need pooling atm.
    # With this config, it always keeps one connection open/ready and could/would use more if multiple are need at once.
    # Defaults are too high, we don't need that much, so only use 1 worker for example.
    conn_pool = AsyncConnectionPool(
        connection_string,
        open=False,
        min_size=min_size,
        max_size=max_size,
        num_workers=1,
        connection_class=AsyncConnection[TupleRow],  # needed to make pyright happy, but is already the default
        timeout=10,  # 30s is way too much
        # these kwargs are passed to the connection constructor
        # set how many times a query needs to be seen for it to be prepared on the server side.
        # COPY statements cannot be prepared, so they ignore this setting.
        kwargs=dict(prepare_threshold=prepare_threshold),
    )

    return conn_pool


async def upsert_df(
    conn: psycopg.AsyncConnection,
    df: pd.DataFrame,
    table_name: str,
    conflict_cols: Sequence[str],
    update_cols: Sequence[str],
):
    """
    Copy df into a temp table then upsert into the target (only conflicting rows).
    Usually you want to specify the primary key columns as conflict_col and all the value columns as update_cols.
    Columns that are not mentioned in the update_cols will stay the same value and not change on conflict.
    """
    # this fancy method is something claude came up with, and I'm glad it did. I'm not that deep in the postgres game.
    temp_name = f"_tmp_{table_name}"
    await conn.execute(
        sql.SQL("CREATE TEMP TABLE {tmp} (LIKE {table} INCLUDING DEFAULTS) ON COMMIT DROP").format(
            tmp=sql.Identifier(temp_name), table=sql.Identifier(table_name)
        )
    )

    # copy data into temp table
    await copy_from_df(conn, df, temp_name)

    # upsert into actual table
    # "EXCLUDED." references the row that would have been inserted but led to the conflict.
    # just "col" or "table.col" would be the value that is already there in the existing row.
    update_set = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(col)) for col in update_cols
    )
    conflict_set = sql.SQL(", ").join(sql.Identifier(c) for c in conflict_cols)
    await conn.execute(
        sql.SQL("INSERT INTO {table} SELECT * FROM {tmp} ON CONFLICT ({conflicts}) DO UPDATE SET {updates}").format(
            table=sql.Identifier(table_name),
            tmp=sql.Identifier(temp_name),
            conflicts=conflict_set,
            updates=update_set,
        )
    )
