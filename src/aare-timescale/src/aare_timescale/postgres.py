from io import BytesIO
from typing import LiteralString, cast

import pandas as pd
from pandas._typing import WriteBuffer
from psycopg import AsyncConnection, sql
from psycopg.abc import Params
from psycopg.rows import TupleRow
from psycopg_pool import AsyncConnectionPool


async def copy_to_df(
    conn: AsyncConnection, query: LiteralString | sql.SQL | sql.Composed, params: Params | None = None
) -> pd.DataFrame:
    """Copy a postgres query into a pandas DataFrame."""
    if isinstance(query, str):
        query = sql.SQL(query)
    async with conn.cursor() as cur:
        with BytesIO() as bio:
            # make sure the query doesn't end with ; somehow, maybe it's automatic?
            async with cur.copy(
                sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query), params
            ) as copy:
                async for data in copy:
                    bio.write(data)
            bio.seek(0)

            return pd.read_csv(bio)


async def copy_from_df(conn: AsyncConnection, df: pd.DataFrame, table: str):
    """Copy a pandas DataFrame into a Timescale table."""
    # save dataframe to an in-memory buffer
    buffer = BytesIO()
    df.to_csv(cast(WriteBuffer[bytes], buffer), index=False, header=True)  # pyright: ignore[reportInvalidCast]

    buffer.seek(0)
    csv = buffer.getvalue()

    async with conn.cursor() as cur:
        # for whatever ungodly reason, it will raise 'invalid input syntax for type timestamp with time zone'
        # if you omit the HEADERs, even though they are supposedly ignored completely, but idk man...
        async with cur.copy(
            sql.SQL("COPY {table} FROM STDIN WITH CSV HEADER").format(table=sql.Identifier(table))
        ) as copy:
            await copy.write(csv)


def init_db_pool(
    connection_string: str, min_size=1, max_size=4, prepare_threshold: int | None = 3
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
