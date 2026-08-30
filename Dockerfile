FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
COPY src /app/src

RUN useradd --create-home --uid 10001 causalcell \
    && mkdir -p /state \
    && chown causalcell:causalcell /state

USER causalcell
ENTRYPOINT ["python", "-m", "causal_cell.repository_pilot"]
