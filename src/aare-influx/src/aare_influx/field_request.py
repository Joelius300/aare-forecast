from typing import cast, Literal

from aare.locations import translate_location


class FieldRequest:
    """
    A request for a field from the InfluxDB. Has a string rep:

    hydro/temperature:first_1h@bern

    smn/rr:sum_1d@thun
    """

    def __init__(
        self,
        measurement: str,
        field: str,
        freq: str,
        agg_fn: str,
        location: str | int,
    ):
        self.measurement = measurement
        self.field = field
        self.freq = freq
        self.agg_fn = agg_fn
        self.location_orig = location
        assert measurement in ["hydro", "smn"], f"Invalid measurement: '{measurement}'"
        self.location = translate_location(location, cast(Literal["hydro", "smn"], measurement))

    def __str__(self):
        return f"{self.measurement}/{self.field}:{self.agg_fn}_{self.freq}@{self.location_orig}"

    def __repr__(self):
        return f"FieldRequest{{{self}}}"

    def __key(self):
        return self.measurement, self.field, self.freq, self.agg_fn, self.location

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, FieldRequest):
            return self.__key() == other.__key()
        return NotImplemented

    @property
    def name(self):
        # need to update if this leads to collisions
        return f"{self.field}_{self.location_orig}"
        # return re.sub(r"[-@/:]", "_", str(self))

    @classmethod
    def from_str(cls, value: str):
        meas_field, agg_freq = value.split(":")
        measurement, field = meas_field.split("/")
        agg_fn, freq_loc = agg_freq.split("_")
        freq, loc = freq_loc.split("@")

        return FieldRequest(measurement, field, freq, agg_fn, loc)
