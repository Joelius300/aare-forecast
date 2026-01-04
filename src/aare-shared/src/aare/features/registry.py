import re
from collections.abc import Callable, Sequence
from typing import Any, cast, overload

from aare.features.rainfall import Rainfall
from aare.features.relative_humidity import RelativeHumidity
from aare.features.wind import Wind
from aare.locations import LOC_ALIAS
import numpy as np
from darts.dataprocessing.transformers import Mapper

from aare.features.air_temp import AirTemp
from aare.features.base.feature import Feature
from aare.features.base.transformed_feature import MapperFuncType, TransformedFeature
from aare.features.flow import Flow
from aare.features.sunshine import Sunshine
from aare.features.water_temp import WaterTemp


def _get_ma_transform(n: str) -> MapperFuncType:
    return lambda ts: ts.window_transform(
        {
            "function": "mean",
            "mode": "rolling",
            "window": int(n),
            "center": False,
        },
        keep_names=True,
    )


class FeatureRegistry:
    """
    Stateless feature registry to create feature instances by name (including suffixes).
    Use like a dict with [] or call get_many(...).
    """

    def __init__(self, additional_features: list[type[Feature]] | None = None):
        self._supported_features: list[type[Feature]] = [
            WaterTemp,
            Flow,
            AirTemp,
            Sunshine,
            Rainfall,
            RelativeHumidity,
            Wind,
        ]

        if additional_features:
            self._supported_features.extend(additional_features)

        self._supported_locations = set([loc.lower() for loc in LOC_ALIAS.keys()])
        self._feature_lookup = {cls.base_name(): cls for cls in self._supported_features}

        self._simple_transformers: dict[str, MapperFuncType] = {
            "sq": lambda ts: ts**2,
            "cube": lambda ts: ts**3,
            # note: signed sqrt = sign(x) * sqrt(abs(x))
            "sqrt": Mapper(lambda x: np.sign(x) * abs(x) ** 0.5),
            "log": Mapper(lambda x: np.sign(x) * np.log(np.abs(x) + 1)),
            "diff": lambda ts: ts.diff(),
            "abs": lambda ts: abs(ts),
            "neg": lambda ts: -ts,
        }

        self._regex_transformers: dict[str, Callable[[Any], MapperFuncType]] = {
            r"^ma(\d+)$": _get_ma_transform,
        }

    def _get_regex_transformer(self, split: str) -> MapperFuncType | None:
        for pattern, factory in self._regex_transformers.items():
            search = re.search(pattern, split)
            if search:
                return factory(*search.groups())

        return None

    def create_feature(self, specifier: str):
        splits = specifier.split("_")

        transformers: list[tuple[str, MapperFuncType]] = []
        location: str | None = None
        base_feature_name: str | None = None
        for split in reversed(splits):
            if split in self._simple_transformers:
                transformers.append((split, self._simple_transformers[split]))
            elif transformer := self._get_regex_transformer(split):
                transformers.append((split, transformer))
            elif split in self._supported_locations:
                if location:
                    raise ValueError(
                        f"Duplicate location found: '{split}', but already found '{location}' in '{specifier}'"
                    )
                location = split
            elif split in self._feature_lookup:
                if base_feature_name:
                    raise ValueError(
                        f"Duplicate base feature found: '{split}', but already found '{base_feature_name}' in '{specifier}'"
                    )

                base_feature_name = split
            else:
                raise ValueError(
                    f"Could not determine what the part '{split}' is in '{specifier}' (not transformer, location or base feature)"
                )

        if not base_feature_name:
            raise ValueError(f"Missing base feature in '{specifier}'")
        if not location:
            raise ValueError(f"Missing location in '{specifier}'")

        feature_cls = self._feature_lookup[base_feature_name]
        # all classes in this dict can be created with just a location
        feature_maker = cast(Callable[[str | int], Feature], feature_cls)
        feature = feature_maker(location)  # if not, it will raise here
        for name, transformer in reversed(transformers):
            feature = TransformedFeature(feature, f"_{name}", transformer)

        return feature

    def _get_item(self, item: str):
        assert isinstance(item, str), "item is not a str"
        return self.create_feature(item)

    def __getitem__(self, item: str):
        return self._get_item(item)

    @overload
    def get_many(self, feature_keys: None) -> None:
        pass

    @overload
    def get_many(self, feature_keys: Sequence[str]) -> list[TransformedFeature | Feature]:
        pass

    def get_many(self, feature_keys: Sequence[str] | None) -> list[TransformedFeature | Feature] | None:
        """Get a list of features. Equiv to [reg[f] for f in features] but handles None -> None"""
        if not feature_keys:
            return None

        # don't overload getitem because then it's typed as returning a union and that's annoying
        return [self._get_item(key) for key in feature_keys]


FEATURES = FeatureRegistry()
"""FeatureRegistry singleton for convenience as that's the way it's been used when it was just a dict."""
