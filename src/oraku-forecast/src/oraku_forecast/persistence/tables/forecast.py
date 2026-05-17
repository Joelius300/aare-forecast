from typing import override
from psycopg_pool import AsyncConnectionPool

from aare_timescale.timescale_table import TimescaleTable


# TODO one table per forecasted variable (currently only hydro/temperature)
#  - column always named the same e.g. value or derived from variable name -> extendable to quantiles
#  - location as a field
class ForecastTable(TimescaleTable):
    def __init__(self, connection_pool: AsyncConnectionPool):
        super().__init__(connection_pool, "forecast", ["run_ts", "time", "temp_bern"])

    @override
    async def ensure_table_exists(self):
        async with self.connection_pool.connection() as conn:
            await conn.execute(
                # run_ts identifies the run, will be FK to metadata table
                """
                CREATE TABLE IF NOT EXISTS forecast
                (
                    run_ts    timestamptz NOT NULL,
                    time      timestamptz NOT NULL,
                    temp_bern real        NOT NULL,
                    PRIMARY KEY (run_ts, time)
                );
                """
            )

            await self.make_hypertable(conn)
