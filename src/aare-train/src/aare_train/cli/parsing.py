from collections.abc import Sequence
from typing import Any, TypeAlias, cast

from aare_train.fetching.feature_identifiers import FeatureIdentifiers


ParamPrimitive: TypeAlias = int | float | bool | str
ParamValue: TypeAlias = ParamPrimitive | Sequence[ParamPrimitive]


def parse_features(targets: Sequence[str], future: Sequence[str] | None) -> FeatureIdentifiers:
    """Parse feature lists into FeatureIdentifiers."""
    return {
        "targets": targets,
        "future": future or [],
    }


def parse_hyperparameters(args: Sequence[str]) -> dict[str, ParamValue]:
    """Parse hyperparameters from command line arguments in key=value format."""
    hparams: dict[str, Any] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Invalid hyperparameter format: {arg}. Expected key=value")

        key, value = arg.split("=", 1)
        hparams[key] = _parse_value(value)

    return hparams


def _parse_value(value: str) -> ParamPrimitive | Sequence[ParamPrimitive]:
    # try to parse as int, float, bool, or keep as string. if "[...]", will parse as list.
    try:
        if value.lower() in ["true", "false"]:
            return value.lower() == "true"
        elif "." in value:
            return float(value)
        elif value.startswith("[") and value.endswith("]"):
            outs: list[ParamPrimitive] = []
            for item in value[1:-1].split(","):
                val = _parse_value(item)
                assert not isinstance(val, list), "Nesting lists is not allowed"
                outs.append(cast(ParamPrimitive, val))

            return outs
        else:
            return int(value)
    except ValueError:
        return value
