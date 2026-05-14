import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import cast
from typing_extensions import deprecated

import pandas as pd
from influxdb_client import InfluxDBClient  # pyright: ignore [reportPrivateImportUsage]

from aare.constants import TIME
from aare_influx.field_request import FieldRequest
from aare.pd_utils import join_many

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
        # noinspection PyArgumentList
        return _chain_equality(column, *values[0], separator, wrap_in_quotes)

    q = '"' if wrap_in_quotes else ""
    return "(r) => " + (f" {separator} ".join([f'r["{column}"] == {q}{value}{q}' for value in values]))


Period = str | datetime | tuple[str | datetime, str | datetime]
Locations = str | int | list[str | int] | None


def _rename_col_after_pivot(df: pd.DataFrame, fields: list[FieldRequest]):
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
    ):
        start, stop = _normalize_period(period)

        return f"""baseData = () =>
    from(bucket: "existenzApi")
        |> range(start: {start}, stop: {stop})
        
getField = (tables=<-, measurement, field, loc) =>
    tables
        |> filter(fn: (r) => r._measurement == measurement and r._field == field and r.loc == loc)
        
postProc = (tables=<-) =>
    tables
        |> pivot(rowKey: ["_time"], columnKey: ["_field", "loc"], valueColumn: "_value")
        |> drop(columns: ["_start", "result", "_stop", "table", "_measurement"])

"""

    @staticmethod
    def _ma(period: str):
        return f"|> timedMovingAverage(every: freq, period: {period})"

    @staticmethod
    def _get_resampler(field: FieldRequest):
        if field.agg_fn == "exact":
            # if field.freq != "1h":
            #     raise ValueError("Only supporting 'exact' for 1h atm")
            # !! requires 'import "date"'
            # return "filter(fn: (r) => date.minute(t: r._time) == 0)"
            raise NotImplementedError("Currently not in use, so disallowed :)")

        agg_sampler = f"aggregateWindow(fn: {field.agg_fn}, every: {field.freq}, createEmpty: false"
        # if we agg with first, we want the 10:00 to take 10:00 or 10:10 or ... but by default it would be
        # 11:00 takes 10:00, 10:10, ... so it needs timeSrc. for last and mean, it will agg >=start, <end
        # so 11:00 contains 10:00, 10:10, ..., 10:50 and no more. the point at 11:00 belongs to bucket 11:00-11:50
        if field.agg_fn == "first":
            agg_sampler += ', timeSrc: "_start"'
        agg_sampler += ")"

        return agg_sampler

    def _build_fields_query(
        self,
        period: Period,
        fields: list[FieldRequest],
    ):
        query = self._base_query(period)
        for field in fields:
            resampler = self._get_resampler(field)
            query += (
                f"{field.name} = baseData() "
                f'|> getField(measurement: "{field.measurement}", field: "{field.field}", loc: "{field.location}")'
                f"|> {resampler}"
                f'|> postProc() |> yield(name: "{field.name}")\n'
            )
        # does yield have significant negative performance implications compared to union? -> couldn't find any yet.

        return query

    def _query(
        self,
        query: str,
        keep_loc: bool,
        locations: Locations,
        fields: list[FieldRequest] | None,
        return_empty: bool,
    ):
        # this function has to accommodate query_hydro and query, which is kinda awkward. could use a refactor.
        df = self.query_raw(query)
        cols = df.columns if isinstance(df, pd.DataFrame) else df[0].columns
        assert not (fields is not None and keep_loc), "keep_loc=True in a field context, unsupported"

        if len(cols) == 0:
            if not return_empty:
                raise ValueError(f"Got no data from influxdb: {df}")

            assert fields is not None, "return_empty=True in a non-field context, unsupported"
            cols = [TIME] + [f.name for f in fields]

            return pd.DataFrame(columns=cols)

        unnecessary_cols = ["result", "table"]
        if (
            not keep_loc
            and "loc" in cols
            and (locations is None or isinstance(locations, int) or isinstance(locations, str) or len(locations) == 1)
        ):
            unnecessary_cols.append("loc")

        if isinstance(df, list):
            df = join_many(*(d.drop(unnecessary_cols, axis=1) for d in df), on=TIME)
        else:
            df = df.drop(unnecessary_cols, axis=1)

        if fields is None:
            # pivoting is done on field and loc, so the result will always contain only that. must manually rename.
            return df

        return _rename_col_after_pivot(df, fields)

    @deprecated("Prefer query() with FieldRequest; this gives you hourly means by default!")
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

        return self._query(query, keep_loc, locations, fields=None, return_empty=False)

    def query(
        self,
        period: Period,
        fields: str | FieldRequest | Iterable[str | FieldRequest],
        return_empty: bool = False,
    ):
        """
        Query a list of fields from the influxdb for the specified period.

        Will raise if influx does not return any data. Set return_empty=True to instead get an empty DataFrame with
        possibly wrong columns (schema mismatch between success and failure).
        """
        if isinstance(fields, str) or isinstance(fields, FieldRequest):
            fields = [fields]

        assert isinstance(fields, Iterable), "fields should be iterable now"
        requests = [field if isinstance(field, FieldRequest) else FieldRequest.from_str(field) for field in fields]
        query = self._build_fields_query(period, requests)

        return self._query(query, keep_loc=False, locations=None, fields=requests, return_empty=return_empty)

    def query_raw(self, query: str) -> pd.DataFrame | list[pd.DataFrame]:
        logger.debug("Executing Flux Query:\n{%s}", query)
        return cast(pd.DataFrame | list[pd.DataFrame], self.client.query_api().query_data_frame(query))  # pyright: ignore[reportUnknownMemberType]
