# CommitRad: the API and the React UI it serves, on port 8000.

# React UI, built once and copied into the Python image (served by the API at /).
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM python:3.11-slim

# git: PyDriller and SZZ (git blame) shell out to it, and the API clones repos.
# libgomp1: OpenMP runtime that LightGBM links against (absent from slim images).
# uv comes from PyPI (pinned) so the build works wherever PyPI is reachable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.11.32

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), exactly as pinned in uv.lock, without dev tools.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY config ./config
COPY models ./models
COPY demo_cache ./demo_cache
COPY --from=web /web/dist ./web/dist
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app/data
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-m", "commitrad.healthcheck"]

CMD ["commitrad", "serve", "--host", "0.0.0.0", "--port", "8000"]
