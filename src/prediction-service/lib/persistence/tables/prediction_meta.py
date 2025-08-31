from datetime import datetime

from psycopg import sql
from psycopg_pool import ConnectionPool

from lib.persistence.timescale_table import TimescaleTable


class PredictionMetaTable(TimescaleTable):
    COLUMNS = [
        "run_ts",
        "model_name",
        "model_version",
        "status",
        "finished_at",
        "horizon",
        "mlflow_run_name",
        "mlflow_exp_id",
        "mlflow_run_id",
        "features_targets",
        "features_past",
        "features_future",
        "error",
    ]

    def __init__(self, connection_pool: ConnectionPool):
        super().__init__(connection_pool, "prediction_meta", self.COLUMNS)

    def ensure_table_exists(self):
        with self.connection_pool.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_meta
                (
                    run_ts              timestamptz   NOT NULL,
                    model_name          text          NOT NULL,
                    model_version       text          NOT NULL,
                    status              text          NOT NULL,
                    finished_at         timestamptz,
                    horizon             smallint      NOT NULL,
                    mlflow_run_name     text          NOT NULL,
                    mlflow_exp_id       text          NOT NULL,
                    mlflow_run_id       text          NOT NULL,
                    features_targets    text          NOT NULL,
                    features_past       text,
                    features_future     text,
                    error               text,
                    PRIMARY KEY (run_ts)
                );
                """
            )

    def update_metadata(self, run_ts: datetime, status: str, error: str | None, finished_at: datetime):
        """
        Update an existing metadata entry with a new status, error and finished_at timestamp, identified by run_ts.
        """
        with self.connection_pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    """
                UPDATE {table}
                SET status = {status}, error = {error}, finished_at = {finished_at}
                WHERE run_ts = {run_ts}
                """
                ).format(
                    table=sql.Identifier(self.table_name),
                    status=status,
                    error=error,
                    run_ts=run_ts,
                    finished_at=finished_at,
                )
            )
