import argparse
from datetime import datetime
import logging

from dataclasses import dataclass
from pathlib import Path

import configargparse

logger = logging.getLogger(__name__)


@dataclass
class CliArgs:
    connection_string: str
    model_path: str
    horizon: int
    simulate_runts: datetime | None
    num_samples: int
    logging_level: str
    loki_url: str | None
    loki_password: str | None


def parse_cli_args() -> CliArgs:
    """Parse CLI args, read ORAKU_* env variables and consider dev_config.yaml for config vars."""
    default_config_file = Path(__file__).parent.parent.parent / "dev_config.yaml"
    p = configargparse.ArgParser(auto_env_var_prefix="oraku_", default_config_files=[default_config_file])
    p.add_argument("-c", "--connection-string", required=True, type=str, help="Connection string for the timescale db")
    p.add_argument("-m", "--model-path", required=True, type=str, help="Path to the model meta file (json)")
    p.add_argument("-n", "--horizon", default=96, type=int, help="Number of hours to forecast into the future")
    p.add_argument(
        "-r",
        "--simulate-runts",
        default=None,
        type=datetime.fromisoformat,
        help="Simulate a run at run_ts. Will pull external data (covariates) from postgres db it usually saves to. Will be stored in TODO??",
    )
    p.add_argument("--num-samples", default=128, type=int, help="Number of samples to take for probabilistic forecasts")
    p.add_argument("--logging-level", default="INFO", type=str, help="Logging level for logging module")
    p.add_argument("--loki-url", default=None, type=str, help="Base URL for the loki instance")
    p.add_argument("--loki-password", default=None, type=str, help="Password for the 'loki' user in loki")

    args_namespace: argparse.Namespace = p.parse_args()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    args = CliArgs(**vars(args_namespace))  # pyright: ignore[reportAny]
    if args.simulate_runts is not None and args.simulate_runts.tzinfo is None:
        logger.warning("Interpreting simulation run_ts as local time, since no timezone was provided!")
        args.simulate_runts = args.simulate_runts.astimezone()

    return args
