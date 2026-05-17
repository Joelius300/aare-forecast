from typing import override
from psycopg_pool import AsyncConnectionPool

from aare_timescale.timescale_table import TimescaleTable


class BafuFlowTable(TimescaleTable):
    def __init__(self, connection_pool: AsyncConnectionPool):
        super().__init__(
            connection_pool,
            "bafu_flow",
            ["run_ts", "time", "location", "last_updated", "flow", "flow_min", "flow_max", "flow_q25", "flow_q75"],
            allow_extra_columns=False,  # we have strict parsing for this source, shouldn't get extras
            allow_missing_columns=True,  # it's acceptable that min, max, q25, or q75 are missing, but not median (flow)
        )

    @override
    async def ensure_table_exists(self):
        async with self.connection_pool.connection() as conn:
            # location is text because it's a categorical and upstream influx also uses string instead of int
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bafu_flow
                (
                    run_ts        timestamptz NOT NULL,
                    time          timestamptz NOT NULL,
                    location      text        NOT NULL,
                    last_updated  timestamptz NOT NULL,
                    flow          real NOT NULL,
                    flow_min      real,
                    flow_max      real,
                    flow_q25      real,
                    flow_q75      real,
                    PRIMARY KEY (run_ts, location, time)
                );
                """
            )

            await self.make_hypertable(conn)

            # make bafu_flow nullable to handle weird cases upstream (e.g. daylight savings)
            await conn.execute("ALTER TABLE bafu_flow ALTER COLUMN flow DROP NOT NULL;")
