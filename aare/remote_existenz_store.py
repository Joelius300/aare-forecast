import logging
from dataclasses import dataclass
from functools import reduce
from typing import cast

import pandas as pd
from influxdb_client import InfluxDBClient  # pyright: ignore [reportPrivateImportUsage]

from aare.constants import TIME

logger = logging.getLogger(__name__)


def _chain_equality(column: str, *values: str | int | list, separator="or", wrap_in_quotes=True):
    """Returns a predicate function where a column is tested against one or more values with equality (==)."""
    # using contains(value: r["loc"], set: ["2135", "2030"]) has muuuuch worse performance
    if values is None or len(values) == 0:
        raise ValueError("cannot equality-chain 0 values")

    if len(values) == 1 and (isinstance(values[0], list)):
        # unpack list so you don't have to on the caller's side
        values: list = cast(list, values[0])

    q = '"' if wrap_in_quotes else ""
    return "(r) => " + (f" {separator} ".join([f'r["{column}"] == {q}{value}{q}' for value in values]))


@dataclass
class FieldRequest:
    """
    A request for a field from the InfluxDB. Has a string rep:

    hydro/temperature:mean_1h

    smn/rr:sum_1d
    """

    measurement: str
    field: str
    freq: str
    agg_fn: str
    # TODO add location with @
    # Add location alias BERN for LOC_BERN and LOC_BERN_SMN, depending on measurement
    # also THUN -> LOC_THUN? etc. not sure what other locations we will want

    # ma: Optional[str] = None

    def __str__(self):
        return f"{self.measurement}/{self.field}:{self.agg_fn}_{self.freq}"

    @property
    def name(self):
        return f"{self.measurement}_{self.field}"

    @classmethod
    def from_str(cls, value: str):
        meas_field, agg_freq = value.split(":")
        measurement, field = meas_field.split("/")
        agg_fn, freq = agg_freq.split("_")

        return FieldRequest(measurement, field, freq, agg_fn)


PERIOD = str | tuple[str, str]
LOCATIONS = str | int | list[str | int]


# TODO column names should be the same as field.name, maybe field_loc
# TODO combine LOC also into column name when pivoting (or additional pivot?)
class RemoteExistenzStore:
    def __init__(self, timeout=60_000, debug=False):
        self.client = InfluxDBClient(
            url="https://influx.konzept.space/",
            # this is a public readonly token, so while not best practice, there's no danger in hard-coding it here :)
            token="0yLbh-D7RMe1sX1iIudFel8CcqCI8sVfuRTaliUp56MgE6kub8-nSd05_EJ4zTTKt0lUzw8zcO73zL9QhC3jtA==",
            org="api.existenz.ch",
            debug=debug,
            timeout=timeout,
        )

    def _base_query(
        self,
        period: str | tuple[str, str],
        locations: str | int | list[str | int],
    ):
        start = period if isinstance(period, str) else period[0]
        stop = "now()" if isinstance(period, str) else period[1]

        return f"""
baseData = () =>
    from(bucket: "existenzApi")
        |> range(start: {start}, stop: {stop})
        |> filter(fn: {_chain_equality("loc", locations)})

getField = (tables=<-, measurement, field, agg_fn, freq=1h) =>
    tables
        |> filter(fn: (r) => r._measurement == measurement and r._field == field)
        |> aggregateWindow(fn: agg_fn, every: freq)
    
postProc = (tables=<-) =>
    tables
        |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> drop(columns: ["_start", "result", "_stop", "table", "_measurement"])

"""

    def _ma(self, period: str):
        return f"|> timedMovingAverage(every: freq, period: {period})"

    def _query_fields(
        self,
        period: PERIOD,
        locations: LOCATIONS,
        fields: list[FieldRequest],
    ):
        query = self._base_query(period, locations)
        for field in fields:
            query += f'{field.name} = baseData() |> getField(measurement: "{field.measurement}", field: "{field.field}", agg_fn: {field.agg_fn}, freq: {field.freq})\n'

        query += "\n"
        query += f"union(tables: [{', '.join((field.name for field in fields))}]) |> postProc()"

        return query

    def _query(
        self,
        query: str,
        keep_loc: bool,
        locations: LOCATIONS,
    ):
        logger.debug("Executing Flux Query:\n{%s}", query)
        df = cast(pd.DataFrame | list[pd.DataFrame], self.client.query_api().query_data_frame(query))

        unnecessary_cols = ["result", "table"]
        if not keep_loc and (isinstance(locations, int) or isinstance(locations, str) or len(locations) == 1):
            unnecessary_cols.append("loc")

        df = reduce(lambda left, right: pd.merge(left, right.drop(unnecessary_cols, axis=1), on=TIME, how="outer"), df)

        return df.drop(unnecessary_cols, axis=1)

    def query_hydro(
        self,
        period: PERIOD,
        locations: LOCATIONS,
        fields: str | list[str] = "temperature",
        agg_freq="1h",  # could also read from params
        agg_func="mean",
        agg_create_empty=False,
        keep_loc=False,
    ):
        """Queries hydrology data from the remote influx store by existenz.ch"""
        # can later be split and extended for non-hydro data
        start = period if isinstance(period, str) else period[0]
        stop = "now()" if isinstance(period, str) else period[1]

        # NOTE: aggregateWindow 1h on 12:00 will take the values from
        #   11:00 until 12:00 and combine them into a single value
        #   with timestamp ** 12:00 **. This means that filtering from
        #   00:00 to xxxx will result in the first entry being
        #   01:00 (aggregate of the values between 00:00 and 01:00).
        #   This might be important because the first entry the model sees
        #   will be 01:00, but it contains values since 00:00 so it's correct.
        #   Can get confusing, esp. if you filter up to but excluding 2024, you
        #   will still get a timestamp on 2024-01-01T00:00:00 containing the last
        #   hour of 2023. IT DOES NOT CONTAIN VALUES FROM 2024!
        # PS. Maybe training data should also be aligned to first of January for consistency.
        #   But it should be enough to align to the stride used in training and validation (prob daily, so 00:00).
        query = (
            f'from(bucket: "existenzApi")\n'
            f"  |> range(start: {start}, stop: {stop})\n"
            f"  |> filter(fn: {_chain_equality('_measurement', 'hydro')})\n"
            f"  |> filter(fn: {_chain_equality('_field', fields)})\n"
            f"  |> filter(fn: {_chain_equality('loc', locations)})\n"
            f"  |> aggregateWindow(every: {agg_freq}, fn: {agg_func}, createEmpty: {str(agg_create_empty).lower()})\n"
            f'  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")\n'
            f'  |> drop(columns: ["_start", "result", "_stop", "table", "_measurement"])'
        )

        return self._query(query, keep_loc, locations)

    def query(
        self,
        period: PERIOD,
        locations: LOCATIONS,
        fields: str | list[str],
        keep_loc=False,
    ):
        requests = [FieldRequest.from_str(field) for field in fields]
        query = self._query_fields(period, locations, requests)

        return self._query(query, keep_loc, locations)
