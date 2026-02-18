import logging
from typing import TypedDict, Literal, cast

from aare.constants import (
    LOC_BERN,
    LOC_BERN_SMN,
    LOC_BRNZ,
    LOC_THUN,
    LOC_THUN_SMN,
    LOC_INT,
    LOC_INT_SMN,
    LOC_BRNZ_SMN,
    LOC_HAGN,
    LOC_HAGN_SMN,
    LOC_BIEL,
    LOC_BIEL_SMN,
    LOC_BRGG,
    LOC_BRGG_SMN,
)

logger = logging.getLogger(__name__)


class LocationIds(TypedDict):
    hydro: int | None
    smn: str | None


# location lookup for all aare guru locations to their respective hydro and smn ids.
# the keys to this dictionary are how features are reference, so the 'bern' in temp_bern (both in the string form
# and in the darts timeseries) comes from here, keep this in mind.
LOC_ALIAS: dict[str, LocationIds] = {
    "BERN": {
        "hydro": LOC_BERN,
        "smn": LOC_BERN_SMN,
    },
    "THUN": {
        "hydro": LOC_THUN,
        "smn": LOC_THUN_SMN,
    },
    "INT": {
        "hydro": LOC_INT,
        "smn": LOC_INT_SMN,
    },
    "BRNZ": {
        "hydro": LOC_BRNZ,
        "smn": LOC_BRNZ_SMN,
    },
    "HAGN": {
        "hydro": LOC_HAGN,
        "smn": LOC_HAGN_SMN,
    },
    "BIEL": {
        "hydro": LOC_BIEL,
        "smn": LOC_BIEL_SMN,
    },
    "BRGG": {
        "hydro": LOC_BRGG,
        "smn": LOC_BRGG_SMN,
    },
}

# inverse of LOC_ALIAS specifically for hydro, so from hydro code (2135) to aare guru location (bern)
LOC_HYDRO_ALIAS: dict[int, str] = {
    cast(int, loc_ids["hydro"]): loc for loc, loc_ids in LOC_ALIAS.items() if loc_ids.get("hydro")
}


def translate_location(loc: str | int, measurement: Literal["hydro", "smn"] | None) -> str:
    """
    Translates a location or alias to the correct location as string.

    "BERN" -> "BER" or "2135" (depending on measurement). 2135 (int) -> "2135".
    """
    assert measurement in [None, "smn", "hydro"], f"bad measurement: '{measurement}'"

    loc = str(loc).upper()
    if loc in LOC_ALIAS:
        if measurement:
            val = LOC_ALIAS[loc][measurement]
            if val is not None:
                return str(val)

            raise ValueError(f"No measurement '{measurement}' for location '{loc}'")
        else:
            raise ValueError(f"Location Alias '{loc}' was passed, but no measurement (hydro or smn) was specified.")

    if measurement:
        logger.warning(
            f"Location '{loc}' is not a known alias and returned verbatim, "
            + f"but measurement '{measurement}' was specified."
        )

    return loc
