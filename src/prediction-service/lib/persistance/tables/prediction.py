from psycopg_pool import ConnectionPool

from lib.persistance.timescale_table import TimescaleTable


class PredictionTable(TimescaleTable):
    def __init__(self, connection_pool: ConnectionPool):
        super().__init__(connection_pool, "prediction", ["run_ts", "time", "temp_bern"])

    def ensure_table_exists(self):
        with self.connection_pool.connection() as conn:
            conn.execute(
                # run_ts identifies the run, will be FK to metadata table
                """
                CREATE TABLE IF NOT EXISTS prediction
                (
                    run_ts    timestamptz NOT NULL,
                    time      timestamptz NOT NULL,
                    temp_bern float       NOT NULL,
                    PRIMARY KEY (run_ts, time)
                );
                """
            )

            self.make_hypertable(conn)
