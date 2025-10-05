from datetime import datetime
from typing import Literal

import pandas as pd
from psycopg import sql
from psycopg_pool import ConnectionPool

from aare.storage.metadata import AareModel
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

    def insert_metadata(self, run_ts: datetime, model_meta: AareModel, config_args, starting_status="started"):
        """
        Store initial information on the run. It should later be updated when the run is finished/crashed.
        """

        def _get_features(cov_type: Literal["past", "future"]):
            # for some reason pycharm is much worse at understanding typings than pyright
            # noinspection PyTypedDict
            features = model_meta["features"].get(cov_type)
            if not features:
                return None

            # noinspection PyTypeChecker
            return ",".join(features)

        # later also add num_samples
        meta = {
            "run_ts": run_ts,
            "model_name": model_meta["name"],
            "model_version": model_meta["version"],
            "status": starting_status,
            "finished_at": None,
            "horizon": config_args.horizon,
            "mlflow_run_name": model_meta["mlflow"]["run_name"],
            "mlflow_exp_id": model_meta["mlflow"]["exp_id"],
            "mlflow_run_id": model_meta["mlflow"]["run_id"],
            "features_targets": ",".join(model_meta["features"]["targets"]),
            "features_past": _get_features("past"),
            "features_future": _get_features("future"),
            "error": None,
        }

        self.insert(pd.DataFrame([meta]))

    def update_metadata(self, run_ts: datetime, status: str, error: str | None, finished_at: datetime):
        """
        Update an existing metadata entry with a new status, error and finished_at timestamp, identified by run_ts.
        """
        with self.connection_pool.connection() as conn:
            conn.execute(
                sql.SQL(
                    """
                UPDATE {table}
                SET status = %(status)s, error = %(error)s, finished_at = %(finished_at)s
                WHERE run_ts = %(run_ts)s
                """
                ).format(
                    table=sql.Identifier(self.table_name),
                ),
                dict(
                    status=status,
                    error=error,
                    run_ts=run_ts,
                    finished_at=finished_at,
                ),
            )
