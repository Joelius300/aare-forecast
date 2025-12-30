from collections.abc import Sequence
from datetime import datetime
from typing import Literal, override

import pandas as pd
from psycopg import sql
from psycopg_pool import AsyncConnectionPool

from aare.storage.metadata import AareModel
from aare_timescale.timescale_table import TimescaleTable

from lib.args import CliArgs


class ForecastMetaTable(TimescaleTable):
    COLUMNS: Sequence[str] = [
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
        "service_version",
    ]

    def __init__(self, connection_pool: AsyncConnectionPool):
        super().__init__(connection_pool, "forecast_meta", self.COLUMNS)

    @override
    async def ensure_table_exists(self):
        async with self.connection_pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_meta
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
            # no hypertable

            await conn.execute("ALTER TABLE forecast_meta ADD COLUMN IF NOT EXISTS service_version text;")

    async def insert_metadata(
        self,
        run_ts: datetime,
        model_meta: AareModel,
        config_args: CliArgs,
        service_version: str,
        starting_status: str = "started",
    ):
        """Store initial information on the run. It should later be updated when the run is finished/crashed."""

        def _get_features(cov_type: Literal["past", "future"]):
            # for some reason pycharm is much worse at understanding typings than pyright
            # noinspection PyTypedDict
            features = model_meta["features"].get(cov_type)
            if not features:
                return None

            # noinspection PyTypeChecker
            return ",".join(features)  # pyright: ignore[reportAny]

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
            "service_version": service_version,
        }

        await self.insert(pd.DataFrame([meta]))

    async def update_metadata(self, run_ts: datetime, status: str, error: str | None, finished_at: datetime):
        """
        Update an existing metadata entry with a new status, error and finished_at timestamp, identified by run_ts.
        """
        async with self.connection_pool.connection() as conn:
            await conn.execute(
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
