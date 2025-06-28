from psycopg_pool import ConnectionPool

from lib.external_sources.meteotest import MeteoTestSource
from lib.persistance.tables.meteotest import MeteotestTable


class SourceRegistry:
    def configure_sources(self, connection_pool: ConnectionPool):
        # TODO can/should read from configs
        return {
            "meteotest": {
                "source": MeteoTestSource("https://aareguru.existenz.ch/rawdata?service=v2018_mdx", ["BERN", "THUN"]),
                "table": MeteotestTable(connection_pool),
            }
        }
