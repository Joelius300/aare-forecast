# AGENTS.md - Aare Forecast Project

This document provides guidance for AI coding agents working in this repository.

## Project Overview

A Python ML project for forecasting Aare river temperature in Bern, Switzerland. Uses time series forecasting with the Darts library, served via FastAPI, with data stored in TimescaleDB.

## Tech Stack

- **Python**: 3.12 (requires >=3.11)
- **Package Manager**: uv (with workspace monorepo)
- **ML Framework**: Darts (time series), PyTorch Lightning
- **API**: FastAPI with uvicorn
- **Database**: TimescaleDB (PostgreSQL extension)
- **Experiment Tracking**: MLflow, Optuna
- **Data Versioning**: DVC

## Project Structure

```
src/
  aare-shared/     # Shared ML library (features, evaluation, storage) for use in service and notebooks (but NOT in API)
  aare-logging/    # Logging utilities with Loki support
  aare-timescale/  # TimescaleDB ORM/utilities
  forecast-api/    # FastAPI REST API for serving forecasts
  forecast-service/ # Service to create forecasts and store them
scripts/           # DVC pipeline scripts
notebooks/         # Jupyter notebooks for exploration and experimentation
reports/           # Marimo analysis notebooks for interested non-technical people
data/              # DVC-tracked data files
```

## Build/Lint/Test Commands

All commands use `just` (command runner) and `uv` (package manager):

```bash
# Setup
just sync              # Install all dependencies (uv sync --all-packages)
just install-kernel    # Install Jupyter kernel

# Linting & Formatting
just lint              # Run ruff check --fix, ruff format, basedpyright
just format            # Run ruff format only

# Running Services
just api               # Run API locally (uvicorn with reload)
just forecast          # Run forecast service

# ML Tools
just mlflow            # Start MLflow UI
just optuna            # Start Optuna dashboard
just track             # Start both mlflow and optuna in parallel

# DVC Pipelines
just repro             # Run DVC pipelines (dvc repro)
just repro -s <stage>  # Run specific DVC stage

# Building Docker Images
just build-service     # Build forecast service Docker image
just build-api         # Build API Docker image
just build             # Build all images in parallel
```

### Running Single Tests

**No formal test suite exists.** Testing is done through:

- Jupyter notebooks for exploration
- DVC pipelines for reproducible evaluation
- MLflow for experiment tracking

TODO comments indicate unit tests are planned but not yet implemented.

## Code Style Guidelines

### General

- Line length: **120 characters** (configured in ruff)
- Comments are only necessary when something is not obvious, and they clearly explain the _why_
- Comments should start with a lowercase letter, not uppercase

### Import Conventions

```python
# 1. Standard library (grouped logically)
from datetime import datetime, timedelta, UTC
from typing import Annotated, cast, Any
from collections.abc import Sequence, Awaitable
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

# 2. Third-party imports
import pandas as pd
from darts import TimeSeries
from fastapi import FastAPI, HTTPException, Depends

# 3. Local imports (use relative imports within same package)
from aare.storage.model import load_model
from lib.args import parse_cli_args
```

### Naming Conventions

| Type              | Convention         | Example                                |
| ----------------- | ------------------ | -------------------------------------- |
| Classes           | PascalCase         | `TimescaleTable`, `FeatureSet`         |
| Functions/methods | snake_case         | `get_inference_data`, `fetch_forecast` |
| Constants         | UPPER_SNAKE_CASE   | `TIME`, `META_SUFFIX`                  |
| Private members   | leading underscore | `_fetch_target`, `_name`               |
| Type aliases      | PascalCase         | `ModelType`, `DataTransformers`        |

### Type Annotations

- **Always annotate** function signatures (parameters and return types)
- Use `basedpyright` for type checking (codebase is not yet fully compliant)
- Use modern typing syntax (Python 3.12+): `list[str]` not `List[str]`

```python
def load_model(
    meta_path: str | Path | None = None,
    *,
    name: str | None = None,
    version: str | None = None,
) -> tuple[AareModel, GlobalForecastingModel, DataTransformers | None]:
    ...

# TypedDict for structured dictionaries
class AareModel(TypedDict):
    model_path: str
    scalers_path: str


# use pyright ignore comments sparingly and with reason
args: argparse.Namespace = p.parse_args()  # pyright: ignore[reportUnknownMemberType]
```

### Error Handling

```python
# assertions for internal invariants
assert min_target_lag is not None and min_target_lag < 0

# ValueError for invalid function inputs
if invalid_locs:
    raise ValueError("Unknown locations: " + ", ".join(invalid_locs))

# HTTPException for API errors (FastAPI)
if horizon > settings.maximum_horizon:
    raise HTTPException(400, f"Horizon exceeds maximum of {settings.maximum_horizon}")

# exception handling with proper logging
try:
    await make_forecast(...)
except Exception as e:
    logger.exception("Forecast run failed", exc_info=True)
```

### Async Patterns

```python
# use asynccontextmanager for lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    db_pool = init_db_pool(settings.connection_string)
    await db_pool.open()
    yield {"db_pool": db_pool}
    await db_pool.close()


# use TaskGroup for concurrent operations
async with asyncio.TaskGroup() as tg:
    fetch_task = tg.create_task(source.fetch())
    tg.create_task(table.ensure_table_exists())
```

### Abstract Base Classes

Use ABCs for extensibility patterns:

```python
class Feature(ABC):
    @abstractmethod
    def cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up raw data. Must not mutate input."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> TimeSeries:
        """Transform cleaned data into feature TimeSeries."""
        pass
```

### Configuration

- Services use `configargparse` with env prefix `ORAKU_`
- API uses `pydantic-settings` with env prefix `ORAKU_`
- Dev config stored in `/dev_config.yaml` for service and `/src/forecast-api/.env` for API

## Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:

- `ruff check --fix` and `ruff format` (local hooks)
- `uv-lock` and `uv-export`
- `dvc-pre-commit`

## Common Patterns

### ML Feature Classes

Machine learning features inherit from `Feature` ABC and implement cleanup/transform:

```python
class WaterTemp(SingleFieldFeature):
    NAME = "temp"

    def __init__(self, loc: str | int):
        super().__init__(self.loc_name(loc), FieldRequest("hydro", "temperature", "1h", "mean", loc))
```

### FastAPI Endpoints

```python
@app.get("/forecast", response_model=ForecastPayload)
async def get_forecasts(
    conn: Annotated[AsyncConnection, Depends(open_db)],
    response: Response,
    horizon: Annotated[int, Query(gt=0, le=settings.maximum_horizon)] = settings.default_horizon,
) -> ForecastPayload:
    ...
```

## Environment Variables

All services use `ORAKU_` prefix for configuration:

- `ORAKU_CONNECTION_STRING` - Database connection
- `ORAKU_MODEL_PATH` - Path to model JSON
- `ORAKU_LOGGING_LEVEL` - Log level (DEBUG, INFO, etc.)
