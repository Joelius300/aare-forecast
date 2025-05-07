import numpy as np
from darts.dataprocessing.transformers import Mapper

from aare.features.air_temp_bern import AirTempBern
from aare.features.base.transformed_feature import TransformedFeature
from aare.features.water_temp_bern import WaterTempBern

FEATURES = {
    "temp_bern": WaterTempBern(),
    "tt_bern": AirTempBern(),
    # must ensure that none of the transformations can result in NaN, Inf or anything of the sorts
    "tt_bern_log": TransformedFeature(AirTempBern(), "_log", Mapper(lambda x: np.sign(x) * np.log(np.abs(x) + 1))),
    "tt_bern_cube": TransformedFeature(AirTempBern(), "_cube", lambda x: x**3),
    "tt_bern_sqrt": TransformedFeature(AirTempBern(), "_sqrt", Mapper(lambda x: np.sign(x) * abs(x) ** 0.5)),
}
