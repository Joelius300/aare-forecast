# Forecast service

Scheduled service responsible for creating forecasts using a trained model and storing them in a database.

## Running locally

Make sure a local database is running, i.e. run `docker compose up` in the project root.

### Directly in venv

```bash
just forecast
```

### In docker

Make sure to build the image first with

```bash
just build-service
```

Then copy the model you want to `model_mount` and start the container.

Linux:

```bash
docker run -e ORAKU_CONNECTION_STRING="host=172.17.0.1 dbname=aare_oraku user=postgres password=password" -e ORAKU_LOGGING_LEVEL=DEBUG -e ORAKU_MODEL_PATH="/models/LR-dev/LR-dev.json" --mount type=bind,src=$PWD/model_mount,dst=/models aare-oraku-forecast:latest
```

<https://stackoverflow.com/questions/48546124/what-is-the-linux-equivalent-of-host-docker-internal>

Windows:

```bash
docker run -e ORAKU_CONNECTION_STRING="host=host.docker.internal dbname=aare_oraku user=postgres password=password" -e ORAKU_LOGGING_LEVEL=DEBUG -e ORAKU_MODEL_PATH="/models/LR-dev/LR-dev.json" --mount type=bind,src=$PWD/model_mount,dst=/models aare-oraku-forecast:latest
```
