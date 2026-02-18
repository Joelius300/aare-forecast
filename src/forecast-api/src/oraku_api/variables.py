# keys are external names exposed via api for all forecasted variables that can be requested.
# values are the internal names used for table names, features, etc.
VARIABLES_EXT2INT = {
    "temperature": "temp",
    "flow": "flow",
    # e.g. air_temperature = tt
}

VARIABLES_EXTERNAL = tuple(VARIABLES_EXT2INT.keys())
VARIABLES_INTERNAL = tuple(VARIABLES_EXT2INT.values())
