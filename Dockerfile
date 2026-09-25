# Мухолет — образ linux/amd64. Варианты сборки см. в README (раздел Docker).
# 1) hermetic с сетью на этапе сборки:  docker build --platform linux/amd64 -t muholet:latest .
# 2) полностью офлайн:                  docker build -f Dockerfile.offline -t muholet:latest .
# Запуск: docker run --rm -p 8080:8080 muholet:latest

# ── stage 1: сборка фронтенда ────────────────────────────────────────────────
FROM --platform=linux/amd64 node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
# внутри сборки нет .git (исключён контекстом) — хеш ревизии приходит снаружи
# и попадает в шапку приложения: по нему видно, из какого коммита собран образ
ARG GIT_SHA=""
ENV GIT_SHA=$GIT_SHA
RUN npm run build

# ── stage 2: рантайм python, зависимости только из локальных wheel ───────────
FROM --platform=linux/amd64 python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN mkdir -p data
COPY wheels/ /wheels/
RUN python -m venv /venv \
    && /venv/bin/pip install --no-index --find-links /wheels \
        fastapi "uvicorn[standard]" numpy "pydantic>=2" websockets \
    && rm -rf /wheels
ENV PATH="/venv/bin:$PATH"
COPY navedenie/ navedenie/
COPY data/ data/
COPY --from=web /web/dist web/dist
EXPOSE 8080
CMD ["uvicorn", "navedenie.app:app", "--host", "0.0.0.0", "--port", "8080"]
