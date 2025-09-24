from psycopg_pool import ConnectionPool

from lib.persistence.timescale_table import TimescaleTable


class MeteotestTable(TimescaleTable):
    def __init__(self, connection_pool: ConnectionPool):
        super().__init__(
            connection_pool, "meteotest", ["run_ts", "time", "location", "tt", "ff", "rr", "dd", "rh", "ss"]
        )

    def ensure_table_exists(self):
        with self.connection_pool.connection() as conn:
            conn.execute(
                # run_ts identifies the run, will be FK to metadata table
                """
                CREATE TABLE IF NOT EXISTS meteotest
                (
                    run_ts   timestamptz NOT NULL,
                    time     timestamptz NOT NULL,
                    location text        NOT NULL,
                    tt       float,
                    ff       float,
                    rr       float,
                    dd       float,
                    rh       float,
                    ss       float,
                    PRIMARY KEY (run_ts, location, time)
                );
                """
            )

            self.make_hypertable(conn)
