import numpy as np
from darts.dataprocessing.transformers import Mapper

from aare.features.air_temp_bern import AirTempBern
from aare.features.base.transformed_feature import TransformedFeature
from aare.features.water_temp_bern import WaterTempBern

FEATURES = {
    "temp_bern": WaterTempBern(),
    "tt_bern": AirTempBern(),
    "tt_bern_log": TransformedFeature(AirTempBern(), "_log", Mapper(np.log)),
    "tt_bern_sqrt": TransformedFeature(AirTempBern(), "_sqrt", lambda x: x**0.5),
    "tt_bern_cube": TransformedFeature(AirTempBern(), "_cube", lambda x: x**3),
}
