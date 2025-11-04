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
  uv run pyright

[group('dev')]
format:
  uv run ruff format

[group('ML')]
mlflow:
  uv run mlflow ui

[group('ML')]
optuna:
  uv run optuna-dashboard sqlite:///data/optuna-trials.db

# start both mlflow ui and optuna-dashboard
[group('ML')]
[parallel]
track: mlflow optuna

[group('build')]
build-service tag='latest':
  docker build -f src/forecast-service/Dockerfile . -t aare-oraku-forecast:{{tag}}

[group('build')]
build-api tag='latest':
  docker build -f src/forecast-api/Dockerfile . -t aare-oraku-api:{{tag}}

# build all
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

[parallel]
[group('run')]
forecast:
  uv run src/forecast-service/main.py