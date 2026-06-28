FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip uv \
    && uv sync --no-dev

RUN uv run python -m src.model.train

FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/src ./src
COPY --from=builder /app/src/artifacts ./src/artifacts
COPY --from=builder /app/.venv ./.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
