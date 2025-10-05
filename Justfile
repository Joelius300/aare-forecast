help:
  @just --list

[group('setup')]
sync:
  uv sync --all-packages

[group('setup')]
install-kernel:
  uv run -m ipykernel install --user --name aare-forecast

[group('dev')]
repro:
  uv run dvc repro

[group('dev')]
lint:
  uv run ruff check --fix
  uv run ruff format
  uv run pyright

[group('dev')]
format:
  uv run ruff format

[group('experiment tracking')]
mlflow:
  uv run mlflow ui

[group('experiment tracking')]
optuna:
  uv run optuna-dashboard sqlite:///data/optuna-trials.db

# start both mlflow ui and optuna-dashboard
[group('ML')]
[parallel]
track: mlflow optuna

[group('build')]
build-service:
  docker build -f src/forecast-service/Dockerfile . -t aare-oraku-forecast:latest

[group('build')]
build-api:
  docker build -f src/forecast-api/Dockerfile . -t aare-oraku-api:latest

# build all
[group('build')]
[parallel]
build: build-service build-api

[group('deploy')]
deploy-service:
  ./deploy/deploy-from-local.sh aare-oraku-forecast

[group('deploy')]
deploy-api:
  ./deploy/deploy-from-local.sh aare-oraku-api

# deploy all
[group('deploy')]
[parallel]
deploy: deploy-service deploy-api

[group('run')]
[working-directory: 'src/forecast-api']
api:
  uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload

[parallel]
[group('run')]
forecast:
  uv run src/forecast-service/main.py