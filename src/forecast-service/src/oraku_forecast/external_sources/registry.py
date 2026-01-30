from datetime import timedelta
from typing import TypedDict

from aare.constants import LOC_BERN, LOC_BRGG, LOC_HAGN, LOC_THUN, LOC_BIEL, LOC_INT
from psycopg_pool import AsyncConnectionPool

from aare_timescale.timescale_table import TimescaleTable
from oraku_forecast.external_sources.bafu_flow import BafuFlowSource
from oraku_forecast.external_sources.external_source import ExternalSource
from oraku_forecast.external_sources.meteotest import MeteoTestSource
from oraku_forecast.persistence.tables.bafu_flow import BafuFlowTable
from oraku_forecast.persistence.tables.meteotest import MeteotestTable


class SourceTuple(TypedDict):
    source: ExternalSource
    table: TimescaleTable


Sources = dict[str, SourceTuple]


class SourceRegistry:
    def configure_sources(self, connection_pool: AsyncConnectionPool) -> Sources:
        # TODO can/should read from configs? params is a bad fit because that's tied to the model! own yaml file maybe
        return {
            # the flow forecast source is special because we actually publish this via the forecast API as well, so it's
            # not just treated as (potential) input for the aare oraku model, but also directly as separate forecasts.
            "bafu_flow": {
                "source": BafuFlowSource(
                    "https://www.hydrodaten.admin.ch/plots/q_forecast/{loc}_q_forecast_de.json",
                    [LOC_BERN, LOC_THUN, LOC_INT, LOC_HAGN, LOC_BIEL, LOC_BRGG],
                    timedelta(minutes=5),
                ),
                "table": BafuFlowTable(connection_pool),
            },
            # the meteotest forecasts are btw. also published via the app, but this is already implemented differently,
            # so for the aare oraku, we only use this as model inputs (and future evaluation, so store everything).
            "meteotest": {
                "source": MeteoTestSource(
                    "https://aareguru.existenz.ch/rawdata?service=v2018_mdx",
                    ["BERN", "THUN", "AARAU", "BRIENZ", "BRUGG", "OLTEN", "RINGGENBERG", "SOLOTHURN", "BIELERSEE"],
                ),
                "table": MeteotestTable(connection_pool),
            },
        }
