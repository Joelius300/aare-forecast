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
