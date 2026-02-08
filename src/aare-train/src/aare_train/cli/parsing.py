from collections.abc import Sequence
from typing import Any

from aare_train.fetching.feature_identifiers import FeatureIdentifiers


def parse_features(targets: Sequence[str], future: Sequence[str] | None) -> FeatureIdentifiers:
    """Parse feature lists into FeatureIdentifiers."""
    return {
        "targets": targets,
        "future": future or [],
    }


def parse_hyperparameters(args: Sequence[str]) -> dict[str, Any]:
    """Parse hyperparameters from command line arguments in key=value format."""
    hparams: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid hyperparameter format: {arg}. Expected key=value")

        key, value = arg.split("=", 1)

        # try to parse as int, float, bool, or keep as string
        try:
            if value.lower() in ["true", "false"]:
                hparams[key] = value.lower() == "true"
            elif "." in value:
                hparams[key] = float(value)
            else:
                hparams[key] = int(value)
        except ValueError:
            hparams[key] = value

    return hparams
