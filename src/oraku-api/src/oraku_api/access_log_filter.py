import logging
from typing import override


class AccessLogFilter(logging.Filter):
    @override
    def filter(self, record: logging.LogRecord) -> bool:
        # ignore logs from the /health endpoint. Not sure why the typing is so fucky.
        # https://stackoverflow.com/questions/70809900/python-fastapi-health-check-logs
        return record.args and len(record.args) >= 3 and record.args[2] != "/health"  # pyright: ignore[reportReturnType, reportArgumentType]
