# must be built from project root
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
# bytecode -> faster startup (but larger size?). copy linkmode -> can copy venv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# use python from image
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# copy just uv.lock and pyproject.toml to install deps without our source.
# annoyingly, we need to copy all packages, otherwise it thinks the lock file is outdated.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=src/aare-train/pyproject.toml,target=src/aare-train/pyproject.toml \
    --mount=type=bind,source=src/aare-train/README.md,target=src/aare-train/README.md \
    --mount=type=bind,source=src/aare-logging/pyproject.toml,target=src/aare-logging/pyproject.toml \
    --mount=type=bind,source=src/aare-timescale/pyproject.toml,target=src/aare-timescale/pyproject.toml \
    --mount=type=bind,source=src/forecast-service/pyproject.toml,target=src/forecast-service/pyproject.toml \
    --mount=type=bind,source=src/forecast-service/README.md,target=src/forecast-service/README.md \
    --mount=type=bind,source=src/forecast-api/pyproject.toml,target=src/forecast-api/pyproject.toml \
    --mount=type=bind,source=src/forecast-api/README.md,target=src/forecast-api/README.md \
    uv sync --locked --no-install-project --no-dev --package=aare-oraku-api

# which files/dirs are copied is controlled by the .dockerignore file.
# wouldn't need sources for forecast-service, but they carry almost no weight so whatever.
COPY . /app

# install packages from our code etc. and remove unneccessary (transitive) dependencies of darts
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --package=aare-oraku-api

# must be same base image as builder, otherwise copying venv won't work
FROM python:3.12-slim-bookworm

WORKDIR /app

# copy the application and built venv from the builder
COPY --from=builder --chown=app:app /app /app

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
RUN chmod +x /app/src/forecast-api/entrypoint.sh
COPY src/forecast-api/deploy/app.json /app/app.json

# MUST NOT EXPOSE, otherwise dokku will want to deploy on that port
# EXPOSE 5000

# use entrypoint to respect PORT variable by dokku
CMD ["/app/src/forecast-api/entrypoint.sh"]
# could probably also do this with the procfile, there it would respect $PORT
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"] <- doesn't respect $PORT
