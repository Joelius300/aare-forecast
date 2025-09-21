import re
from collections.abc import Callable

import numpy as np
from darts.dataprocessing.transformers import Mapper

from aare.features.air_temp_bern import AirTempBern
from aare.features.base.feature import Feature
from aare.features.base.transformed_feature import TransformedFeature
from aare.features.flow_bern import FlowBern
from aare.features.sunshine_bern import SunshineBern
from aare.features.water_temp_bern import WaterTempBern


def _make_ma(feature: Feature, n: int):
    return TransformedFeature(
        feature,
        f"_ma{n}",
        lambda ts: ts.window_transform(
            {
                "function": "mean",
                "mode": "rolling",
                "window": n,
                "center": False,
            },
            keep_names=True,
        ),
    )


class FeatureRegistry:
    """Stateless feature registry to create feature instances by name (including suffixes). Use like a dict with []."""

    def __init__(self):
        # lazy lookup -> classes (without init args) or param-less lambdas
        self.lookup: dict[str, Callable[[], Feature]] = {
            "temp_bern": WaterTempBern,
            "tt_bern": AirTempBern,
            "ss_bern": SunshineBern,
            "flow_bern": FlowBern,
            # must ensure that none of the transformations can result in NaN, Inf or anything of the sorts
            "tt_bern_log": lambda: TransformedFeature(
                AirTempBern(), "_log", Mapper(lambda x: np.sign(x) * np.log(np.abs(x) + 1))
            ),
            "tt_bern_cube": lambda: TransformedFeature(AirTempBern(), "_cube", lambda ts: ts**3),
            "tt_bern_sqrt": lambda: TransformedFeature(
                AirTempBern(), "_sqrt", Mapper(lambda x: np.sign(x) * abs(x) ** 0.5)
            ),
        }

    def __getitem__(self, item: str):
        assert isinstance(item, str), "Cannot use registry with something other than string."

        ma_match = re.search(r"^(\w+)_ma(\d+)$", item)
        if ma_match is not None:
            feature = ma_match.group(1)
            ma_len = int(ma_match.group(2))
            return _make_ma(self.lookup[feature](), ma_len)

        return self.lookup[item]()


FEATURES = FeatureRegistry()
"""FeatureRegistry singleton for convenience as that's the way it's been used when it was just a dict."""
