from io import BytesIO
from typing import LiteralString, cast

import pandas as pd
from pandas._typing import WriteBuffer
from psycopg import AsyncConnection, sql
from psycopg.abc import Params


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
