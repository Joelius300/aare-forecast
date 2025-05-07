from aare.features.air_temp_bern import AirTempBern
from aare.features.water_temp_bern import WaterTempBern

FEATURES = {
    "temp_bern": WaterTempBern(),
    "tt_bern": AirTempBern(),
}
