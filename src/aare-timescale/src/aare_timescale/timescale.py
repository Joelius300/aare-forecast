from psycopg import AsyncConnection, sql


async def make_hypertable(conn: AsyncConnection, table_name: str, time_col: str = "time", chunk_interval: int = 7):
    """Call create_hypertable to turn the table into a hypertable IF IT'S NOT ALREADY ONE (idempotent)."""
    await conn.execute(
        # could add second partitioning dimension with add_dimension after create_hypertable
        sql.SQL("""
        SELECT *
        FROM create_hypertable({table}, by_range({time_col}, INTERVAL '{chunk_interval} days'), if_not_exists => TRUE);
        """).format(table=table_name, time_col=time_col, chunk_interval=chunk_interval)
        # strings will be wrapped in ''
    )
