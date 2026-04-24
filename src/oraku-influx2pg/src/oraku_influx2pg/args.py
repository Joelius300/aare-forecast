from dataclasses import dataclass
from pathlib import Path

import configargparse


@dataclass
class CliArgs:
    connection_string: str
    fields: list[str]
    logging_level: str
    loki_url: str | None
    loki_password: str | None


def parse_cli_args() -> CliArgs:
    default_config_file = Path(__file__).parent.parent.parent / "dev_config.yaml"
    p = configargparse.ArgParser(auto_env_var_prefix="oraku_", default_config_files=[default_config_file])
    p.add_argument("-c", "--connection-string", required=True, type=str, help="Connection string for the timescale db")
    p.add_argument(
        "-f",
        "--fields",
        required=True,
        nargs="+",
        type=str,
        help='FieldRequest strings, e.g. "hydro/temperature:first_1h@bern"',
    )
    p.add_argument("--logging-level", default="INFO", type=str, help="Logging level for logging module")
    p.add_argument("--loki-url", default=None, type=str, help="Base URL for the loki instance")
    p.add_argument("--loki-password", default=None, type=str, help="Password for the 'loki' user in loki")

    return CliArgs(**vars(p.parse_args()))  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
