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
build-pred:
  docker build -f src/prediction-service/Dockerfile . -t aare-oraku-prediction:latest

[group('build')]
build-api:
  docker build -f src/prediction-api/Dockerfile . -t aare-oraku-prediction-api:latest

# build all
[group('build')]
[parallel]
build: build-pred build-api

[group('deploy')]
deploy-pred:
  ./deploy/deploy-from-local.sh aare-oraku-prediction

[group('deploy')]
deploy-api:
  ./deploy/deploy-from-local.sh aare-oraku-prediction-api

# deploy all
[group('deploy')]
[parallel]
deploy: deploy-pred deploy-api

[group('run')]
[working-directory: 'src/prediction-api']
api:
  uv run uvicorn main:app --host 0.0.0.0 --port 8080 --reload

[parallel]
[group('run')]
predict:
  uv run src/prediction-service/main.py