from typing import override
from psycopg_pool import AsyncConnectionPool

from aare_timescale.timescale_table import TimescaleTable


class MeteotestTable(TimescaleTable):
    def __init__(self, connection_pool: AsyncConnectionPool):
        super().__init__(
            connection_pool,
            "meteotest",
            ["run_ts", "time", "location", "tt", "ff", "rr", "dd", "rh", "ss"],
            allow_extra_columns=True,
            allow_missing_columns=True,
        )

    @override
    async def ensure_table_exists(self):
        async with self.connection_pool.connection() as conn:
            await conn.execute(
                # run_ts identifies the run, will be FK to metadata table
                """
                CREATE TABLE IF NOT EXISTS meteotest
                (
                    run_ts   timestamptz NOT NULL,
                    time     timestamptz NOT NULL,
                    location text        NOT NULL,
                    tt       real,
                    ff       real,
                    rr       real,
                    dd       real,
                    rh       real,
                    ss       real,
                    PRIMARY KEY (run_ts, location, time)
                );
                """
            )

            await self.make_hypertable(conn)
