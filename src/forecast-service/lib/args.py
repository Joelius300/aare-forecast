import argparse

import configargparse


def get_args() -> argparse.Namespace:
    p = configargparse.ArgParser(auto_env_var_prefix="oraku_", default_config_files=["./dev_config.yaml"])
    p.add_argument(
        "-c", "--connection-string", required=True, type=str, help="Connection string for the postgres database"
    )
    p.add_argument("-m", "--model-path", required=True, type=str, help="Path to the model meta file (json)")
    p.add_argument("-n", "--horizon", default=96, type=int, help="Number of hours to forecast into the future")
    p.add_argument("--num-samples", default=128, type=int, help="Number of samples to take for probabilistic forecasts")
    p.add_argument("--logging-level", default="INFO", type=str, help="Logging level for logging module")
    p.add_argument("--loki-url", default=None, type=str, help="Base URL for the loki instance")
    p.add_argument("--loki-password", default=None, type=str, help="Password for the 'loki' user in loki")
    # TODO typing via vars() either into TypedDict or pydantic

    return p.parse_args()
