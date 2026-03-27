import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from functools import reduce
from typing import cast, Optional

import pandas as pd
from influxdb_client import InfluxDBClient  # pyright: ignore [reportPrivateImportUsage]

from aare.constants import TIME
from aare_influx.field_request import FieldRequest

logger = logging.getLogger(__name__)


def _chain_equality(
    column: str, *values: str | int | Sequence[str | int], separator: str = "or", wrap_in_quotes: bool = True
) -> str:
    """Returns a flux predicate function where a column is tested against one or more values with equality (==)."""
    # using contains(value: r["loc"], set: ["2135", "2030"]) has muuuuch worse performance
    if len(values) == 0:
        raise ValueError("cannot equality-chain 0 values")

    # must check for list or tuple here because strings are also sequences
    if len(values) == 1 and (isinstance(values[0], (list, tuple))):
        # unpack list so you don't have to on the caller's side
        return _chain_equality(column, *values[0], separator, wrap_in_quotes)

    q = '"' if wrap_in_quotes else ""
    return "(r) => " + (f" {separator} ".join([f'r["{column}"] == {q}{value}{q}' for value in values]))


Period = str | datetime | tuple[str | datetime, str | datetime]
Locations = str | int | list[str | int] | None


def _rename_col_after_pivot(df: pd.DataFrame, fields: Optional[list[FieldRequest]]):
    # pivoting is done on field and loc, so the result will always contain only that. must manually rename.
    if fields is None:
        return df

    mapper = {f"{field.field}_{field.location}": field.name for field in fields}
    return df.rename(mapper, axis="columns", errors="raise")


def _normalize_date(date: str | datetime) -> str:
    if isinstance(date, str):
        return date

    assert isinstance(date, datetime) and date.tzinfo is not None, (
        "Must use string or datetime-aware datetime to query influxdb"
    )
    return date.isoformat()


def _normalize_period(period: Period) -> tuple[str, str]:
    if isinstance(period, (str, datetime)):
        # single str or datetime means 'from that point to now'
        return _normalize_date(period), "now()"

    assert isinstance(period, tuple) and len(period) == 2, "Invalid period format"
    return _normalize_date(period[0]), _normalize_date(period[1])


class RemoteExistenzStore:
    """Client for fetching data from the remote aare.guru InfluxDB database."""

    def __init__(self, timeout: int = 60_000, debug: bool = False):
        self.client: InfluxDBClient = InfluxDBClient(
            url="https://influx.konzept.space/",
            # this is a public readonly token, so while not best practice, there's no danger in hard-coding it here :)
            token="0yLbh-D7RMe1sX1iIudFel8CcqCI8sVfuRTaliUp56MgE6kub8-nSd05_EJ4zTTKt0lUzw8zcO73zL9QhC3jtA==",
            org="api.existenz.ch",
            debug=debug,
            timeout=timeout,
        )

    @staticmethod
    def _base_query(
        period: Period,
        locations: Locations,
    ):
        start, stop = _normalize_period(period)

        loc_filter = "" if not locations else f"|> filter(fn: {_chain_equality('loc', locations)})"

        return f"""baseData = () =>
    from(bucket: "existenzApi")
        |> range(start: {start}, stop: {stop})
        {loc_filter}

getField = (tables=<-, measurement, field, agg_fn, loc, freq=1h) =>
    tables
        |> filter(fn: (r) => r._measurement == measurement and r._field == field and r.loc == loc)
        |> aggregateWindow(fn: agg_fn, every: freq, createEmpty: false)

postProc = (tables=<-) =>
    tables
        |> pivot(rowKey: ["_time"], columnKey: ["_field", "loc"], valueColumn: "_value")
        |> drop(columns: ["_start", "result", "_stop", "table", "_measurement"])

"""

    @staticmethod
    def _ma(period: str):
        return f"|> timedMovingAverage(every: freq, period: {period})"

    def _query_fields(
        self,
        period: Period,
        fields: list[FieldRequest],
        locations: Locations = None,
    ):
        query = self._base_query(period, locations)
        for field in fields:
            query += (
                f"{field.name} = baseData() "
                f'|> getField(measurement: "{field.measurement}", field: "{field.field}",'
                f' loc: "{field.location}", agg_fn: {field.agg_fn}, freq: {field.freq}) '
                f'|> postProc() |> yield(name: "{field.name}")\n'
            )
        # does yield have significant negative performance implications compared to union? -> couldn't find any yet.

        return query

    def _query(
        self,
        query: str,
        keep_loc: bool,
        locations: Locations = None,
        fields: list[FieldRequest] | None = None,
    ):
        logger.debug("Executing Flux Query:\n{%s}", query)
        df = cast(pd.DataFrame | list[pd.DataFrame], self.client.query_api().query_data_frame(query))
        cols = df.columns if isinstance(df, pd.DataFrame) else df[0].columns
        if len(cols) == 0:
            raise ValueError(f"Got no data from influxdb: {df}")

        unnecessary_cols = ["result", "table"]
        if (
            not keep_loc
            and "loc" in cols
            and (locations is None or isinstance(locations, int) or isinstance(locations, str) or len(locations) == 1)
        ):
            unnecessary_cols.append("loc")

        if isinstance(df, list):
            df = reduce(
                lambda left, right: pd.merge(left, right.drop(unnecessary_cols, axis=1), on=TIME, how="outer"), df
            )

        return _rename_col_after_pivot(df.drop(unnecessary_cols, axis=1), fields)

    def query_hydro(
        self,
        period: Period,
        locations: str | int | list[str | int],
        fields: str | list[str] = "temperature",
        agg_freq: str = "1h",  # could also read from params
        agg_func: str = "mean",
        agg_create_empty: bool = False,
        keep_loc: bool = False,
    ):
        """Queries hydrology data from the remote influx store by existenz.ch"""
        # can later be split and extended for non-hydro data
        start, stop = _normalize_period(period)

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
        period: Period,
        fields: str | Iterable[str | FieldRequest],
        keep_loc: bool = False,
    ):
        if isinstance(fields, str) or isinstance(fields, FieldRequest):
            fields = [fields]

        requests = [field if isinstance(field, FieldRequest) else FieldRequest.from_str(field) for field in fields]
        query = self._query_fields(period, requests)

        return self._query(query, keep_loc, fields=requests)
