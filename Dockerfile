# Build stage — installe les dépendances avec uv
FROM python:3.11-slim AS builder

WORKDIR /app

# Installe uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copie les fichiers de dépendances uniquement (cache layer)
COPY pyproject.toml uv.lock ./

# Installe les dépendances dans un venv isolé
RUN uv sync --frozen --no-dev --no-install-project


# Runtime stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copie le venv depuis le builder
COPY --from=builder /app/.venv /app/.venv

# Met le venv en tête de PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copie le code source, les modèles et les données
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Ports exposés (API=8000, Dashboard=8501)
EXPOSE 8000 8501
