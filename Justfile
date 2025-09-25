help:
  @just --list

sync:
  uv sync --all-packages

install-kernel:
  uv run -m ipykernel install --user --name aare-forecast

repro:
  uv run dvc repro

lint:
  uv run ruff check --fix
  uv run ruff format
  uv run pyright

format:
  uv run ruff format

mlflow:
  uv run mlflow ui

optuna:
  uv run optuna-dashboard sqlite:///data/optuna-trials.db

# start both mlflow ui and optuna-dashboard. on my setup, ctrl+c once closes both, idk..
track:
  just mlflow & just optuna && fg

build-pred:
  docker build -f src/prediction-service/Dockerfile . -t aare-oraku-prediction:latest

build-api:
  docker build -f src/prediction-api/Dockerfile . -t aare-oraku-prediction-api:latest

build:
    just build-pred
    just build-api
