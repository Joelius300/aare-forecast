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


FEATURES = {
    "temp_bern": WaterTempBern(),
    "tt_bern": AirTempBern(),
    # must ensure that none of the transformations can result in NaN, Inf or anything of the sorts
    "tt_bern_log": TransformedFeature(AirTempBern(), "_log", Mapper(lambda x: np.sign(x) * np.log(np.abs(x) + 1))),
    "tt_bern_cube": TransformedFeature(AirTempBern(), "_cube", lambda ts: ts**3),
    "tt_bern_sqrt": TransformedFeature(AirTempBern(), "_sqrt", Mapper(lambda x: np.sign(x) * abs(x) ** 0.5)),
    "tt_bern_ma3": _make_ma(AirTempBern(), 3),
    "tt_bern_ma6": _make_ma(AirTempBern(), 6),
    "tt_bern_ma12": _make_ma(AirTempBern(), 12),
    "tt_bern_ma24": _make_ma(AirTempBern(), 24),
    "tt_bern_ma60": _make_ma(AirTempBern(), 60),
    "ss_bern": SunshineBern(),
    "ss_bern_ma3": _make_ma(SunshineBern(), 3),
    "ss_bern_ma6": _make_ma(SunshineBern(), 6),
    "ss_bern_ma12": _make_ma(SunshineBern(), 12),
    "ss_bern_ma24": _make_ma(SunshineBern(), 24),
    "ss_bern_ma60": _make_ma(SunshineBern(), 60),
    "flow_bern": FlowBern(),
}
