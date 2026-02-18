help:
  @just --list

[group('setup')]
sync:
  uv sync --all-packages

[group('setup')]
install-kernel:
  uv run -m ipykernel install --user --name aare-forecast

[group('dev')]
repro *FLAGS:
  uv run dvc repro {{FLAGS}}

[group('dev')]
lint:
  uv run ruff check --fix
  uv run ruff format
  uv run basedpyright --level error

[group('dev')]
format:
  uv run ruff format

[group('ML')]
mlflow:
  uv run mlflow ui --backend-store-uri sqlite:///mlruns.db

[group('ML')]
optuna:
  uv run optuna-dashboard sqlite:///data/optuna-trials.db

# start both mlflow ui and optuna-dashboard
[group('ML')]
[parallel]
track: mlflow optuna

[group('build')]
build-service tag='latest':
  -uv version --package aare-oraku-forecast {{tag}}  # try setting version, ignore if failed
  docker build -f src/forecast-service/Dockerfile . -t aare-oraku-forecast:{{tag}} -t aare-oraku-forecast:latest

[group('build')]
build-api tag='latest':
  -uv version --package aare-oraku-api {{tag}}  # try setting version, ignore if failed
  docker build -f src/forecast-api/Dockerfile . -t aare-oraku-api:{{tag}} -t aare-oraku-api:latest

[group('build')]
[working-directory: 'reports']
build-report:
  uv run marimo export html-wasm model-eval.py -o model-eval-wasm-notebook --mode run

# build all docker images
[group('build')]
[parallel]
build: build-service build-api

[group('deploy')]
deploy-service tag='latest':
  ./deploy/deploy-from-local.sh aare-oraku-forecast {{tag}}

[group('deploy')]
deploy-api tag='latest':
  ./deploy/deploy-from-local.sh aare-oraku-api {{tag}}

# deploy latest of service and api
[group('deploy')]
[parallel]
deploy-latest: deploy-service deploy-api

[group('run')]
[working-directory: 'src/forecast-api']
api:
  uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload

[group('run')]
forecast:
  uv run src/forecast-service/main.py

# run forecast service via docker (only works on linux, with the docker compose running, and a model in model_mount) [set model with -m or --model]
[group('run')]
[arg("model", short="m", long)]
forecast-docker tag='latest' model='LR-dev':
    docker run --env ORAKU_CONNECTION_STRING="host=172.17.0.1 dbname=aare_oraku user=postgres password=password" --env ORAKU_MODEL_PATH="/models/{{model}}.json" --env ORAKU_LOGGING_LEVEL="DEBUG" --env PYTHONUNBUFFERED=1 --mount type=bind,src=$PWD/model_mount,dst=/models --rm aare-oraku-forecast:{{tag}}

# run forecast api via docker (only works on linux, with the docker compose running) [port with -p]
[group('run')]
[arg("port", short="p")]
api-docker tag='latest' port='5000':
  docker run --env ORAKU_CONNECTION_STRING="host=172.17.0.1 dbname=aare_oraku user=postgres password=password" --env ORAKU_LOGGING_LEVEL="DEBUG" -p 8080:{{port}} --rm aare-oraku-api:{{tag}}
