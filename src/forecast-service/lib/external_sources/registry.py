from typing import TypedDict

from psycopg_pool import AsyncConnectionPool

from aare_timescale.timescale_table import TimescaleTable
from lib.external_sources.external_source import ExternalSource
from lib.external_sources.meteotest import MeteoTestSource
from lib.persistence.tables.meteotest import MeteotestTable


class SourceTuple(TypedDict):
    source: ExternalSource
    table: TimescaleTable


Sources = dict[str, SourceTuple]


class SourceRegistry:
    def configure_sources(self, connection_pool: AsyncConnectionPool) -> Sources:
        # TODO can/should read from configs
        return {
            "meteotest": {
                "source": MeteoTestSource("https://aareguru.existenz.ch/rawdata?service=v2018_mdx", ["BERN", "THUN"]),
                "table": MeteotestTable(connection_pool),
            }
        }
