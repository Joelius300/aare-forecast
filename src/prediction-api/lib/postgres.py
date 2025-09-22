from io import BytesIO

import pandas as pd
import psycopg
from psycopg import sql
from datetime import datetime


# duplicate code, I'm a sinner
def copy_to_df(conn: psycopg.Connection, query: sql.SQL):
    """Copy a Timescale query into a pandas DataFrame."""
    with conn.cursor() as cur:
        with BytesIO() as bio:
            with cur.copy(sql.SQL("COPY ({query}) TO STDOUT WITH CSV HEADER").format(query=query)) as copy:
                for data in copy:
                    bio.write(data)
            bio.seek(0)

            return pd.read_csv(bio)

def get_predictions(at: datetime, horizon: int):
    query = sql.SQL("""
    select run_ts, time, temp_bern from predictions
    """)
    df = copy_to_df()
    # TODOOOOOO :)
