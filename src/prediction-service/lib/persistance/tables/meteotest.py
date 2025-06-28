from psycopg_pool import ConnectionPool

from lib.persistance.timescale_table import TimescaleTable


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
                    location varchar(8)  NOT NULL,
                    tt       float       NOT NULL,
                    ff       float       NOT NULL,
                    rr       float       NOT NULL,
                    dd       float       NOT NULL,
                    rh       float       NOT NULL,
                    ss       float       NOT NULL,
                    PRIMARY KEY (run_ts, location, time)
                );
                """
            )

            conn.execute(
                # could add second partitioning dimension with add_dimension after create_hypertable
                """
                SELECT *
                FROM create_hypertable('meteotest', by_range('time', INTERVAL '7 days'));
                """
            )
