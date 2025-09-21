from io import BytesIO
from typing import cast

import pandas as pd
import psycopg
from pandas._typing import WriteBuffer
from psycopg import sql


# duplicate code, I'm a sinner
def copy_from_df(conn: psycopg.Connection, df: pd.DataFrame, table: str):
    """Copy a pandas DataFrame into a postgres table."""
    # save dataframe to an in-memory buffer
    buffer = BytesIO()
    df.to_csv(cast(WriteBuffer[bytes], buffer), index=False, header=True)

    buffer.seek(0)
    csv = buffer.getvalue()

    with conn.cursor() as cur:
        # for whatever ungodly reason, it will raise 'invalid input syntax for type timestamp with time zone'
        # if you omit the HEADERs, even though they are supposedly ignored completely, but idk man...
        with cur.copy(sql.SQL("COPY {table} FROM STDIN WITH CSV HEADER").format(table=sql.Identifier(table))) as copy:
            copy.write(csv)
