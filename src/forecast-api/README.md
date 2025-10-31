# Forecast API

This API serves the stored forecasts from the timescaledb.

I initially wanted to build this with ASP.NET, so I finally get to use it again, but frankly it doesn't make sense.

## Running locally

Make sure a local database is running, i.e. run `docker compose up` in the project root.

### Directly in venv

```bash
just api
```

This uses uvicorn and reloads when files change. \
Ps. I have genuinely no idea why the fastapi-cli (`fastapi dev`) isn't working.

### In docker

Make sure to build the image first with

```bash
just build-api
```

Linux:

```bash
docker run -e ORAKU_CONNECTION_STRING="host=172.17.0.1 dbname=aare_oraku user=postgres password=password" -e ORAKU_LOGGING_LEVEL=DEBUG -p 5001:5000 aare-oraku-api:latest
```

<https://stackoverflow.com/questions/48546124/what-is-the-linux-equivalent-of-host-docker-internal>

Windows:

```bash
docker run -e ORAKU_CONNECTION_STRING="host=host.docker.internal dbname=aare_oraku user=postgres password=password" -e ORAKU_LOGGING_LEVEL=DEBUG -p 5001:5000 aare-oraku-api:latest
```
