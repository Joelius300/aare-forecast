import logging
from typing import TypedDict, Literal, Optional

from aare.constants import LOC_BERN, LOC_BERN_SMN, LOC_THUN, LOC_THUN_SMN, LOC_INT, LOC_INT_SMN

logger = logging.getLogger(__name__)


class LocationIds(TypedDict):
    hydro: int | None
    smn: str | None


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
}


def translate_location(loc: str | int, measurement: Optional[Literal["hydro", "smn"]]) -> str:
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
