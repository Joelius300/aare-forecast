import numpy as np
from darts.dataprocessing.transformers import Mapper

from aare.features.air_temp_bern import AirTempBern
from aare.features.base.transformed_feature import TransformedFeature
from aare.features.water_temp_bern import WaterTempBern


def _make_ma(n: int):
    return TransformedFeature(
        AirTempBern(),
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
    "tt_bern_ma3": _make_ma(3),
    "tt_bern_ma6": _make_ma(6),
    "tt_bern_ma12": _make_ma(12),
    "tt_bern_ma24": _make_ma(24),
    "tt_bern_ma60": _make_ma(60),
}
