FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Ensure Python can always find files in the root directory
ENV PYTHONPATH="/app"

# 1. Copy config and install dependencies only (uses Docker layer caching)
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# 2. Copy the actual source code
COPY --chown=user . .

# 3. Install the project modules and link the entry points
RUN uv sync --frozen --no-cache

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

CMD ["uv", "run", "server"]