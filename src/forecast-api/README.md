# Forecast API

This API serves the stored forecasts from the timescaledb.

I initially wanted to build this with ASP.NET, so I finally get to use it again, but frankly it doesn't make sense.

Ps. I have genuinely no idea why the fastapi-cli (`fastapi dev`) isn't working.
Can run with `uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload` instead.
