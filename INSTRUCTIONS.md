
## 0. Получение исходников

Проект размещён на GitHub:

```bash
git clone https://github.com/UserCommon/diploma.git diploma_fixed
cd diploma_fixed
```

## 1. Необходимые зависимости

### Главный путь (Docker)

| Инструмент | Версия | Назначение |
|------------|--------|------------|
| Docker + Docker Compose | актуальная | Все сервисы (инфраструктура, backend, фронтенд) |

### Дополнительно для альтернативного пути (локальный uv)

| Инструмент | Версия |
|------------|--------|
| Python | 3.11 или 3.12 |
| uv | актуальная (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Node.js | 18+ (рекомендуется 20+) |
| npm | идёт с Node.js |

## 2. Переменные окружения

Для POLZA_AI_API_KEY, прилагаю ключ на 200 рублей генераций: pza_c9TngHDFkhuvtE-fme4dML1RcqM28Bm4

Создать файл `core_service/.env`:

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/core_db
REDIS_URL=redis://localhost:6379
STORAGE_PATH=./storage
CORS_ORIGINS=http://localhost:5173
ACCESSORIES_SERVICE_URL=http://localhost:8000

# LLM-агент
LLM_PROVIDER=openai_compatible
LLM_API_KEY=<твой ключ>
LLM_BASE_URL=https://polza.ai/api/v1
LLM_MODEL=moonshotai/kimi-k2.6

# Провайдер инпейнтинга
INPAINT_PROVIDER=genapi
GENAPI_API_KEY=<твой ключ>
POLZA_AI_API_KEY=<твой ключ>
POLZA_MODEL=black-forest-labs/flux.2-pro
GPT_IMAGE_MODEL=openai/gpt-image-1.5

# Опционально, для FLUX Fill с HuggingFace
HF_TOKEN=<твой токен>
```

В Docker-режиме значения `DATABASE_URL`, `REDIS_URL`, `ACCESSORIES_SERVICE_URL` переопределяются в `docker-compose.yml` на внутренние имена контейнеров — менять их вручную не нужно.


---

## 3. Главный путь: запуск через Docker Compose

Из корня репозитория одной командой поднимается весь стек — инфраструктура, оба backend-сервиса и фронтенд. Профиль `core` включает `core-api` + `core-worker`:

```bash
docker-compose --profile core up -d
```

Что поднимается:

| Контейнер | Порт | Назначение |
|-----------|------|------------|
| `postgres` | 5432 | PostgreSQL 16 + pgvector, БД `accessories_db` и `core_db` |
| `redis` | 6379 | Очередь SAQ |
| `accessories-api` | 8000 | FastAPI для деталей |
| `accessories-worker` | — | SAQ-воркер обработки деталей |
| `core-api` | 8001 | FastAPI ИИ-агента |
| `core-worker` | — | SAQ-воркер агента |
| `frontend` | 5173 | SvelteKit dev-сервер (Vite, hot-reload) |

Проверка состояния:

```bash
docker-compose ps
```

UI: `http://localhost:5173`.

### Остановка

```bash
docker-compose --profile core down
# с очисткой данных БД:
docker-compose --profile core down -v
```

---

## 4. Альтернатива: core_service локально через uv

Этот путь полезен, если нужно отлаживать `core_service` без пересборки контейнера, либо использовать локальные ML-модели на GPU/MPS.

### Шаг 1. Инфраструктура + accessories через Docker

Запускаем только базовые сервисы (без профиля `core`):

```bash
docker-compose up -d
```

Поднимутся: `postgres`, `redis`, `accessories-api`, `accessories-worker`.

### Шаг 2. core_service через uv

```bash
cd core_service
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --port 8001 --reload &
uv run python -m app.workers.worker &
cd ..
```

### Шаг 3. Фронтенд

```bash
cd frontend
npm install
npm run dev
```

### Остановка альтернативного пути

```bash
pkill -f "uvicorn app.main:app"
pkill -f "app.workers.worker"
# фронтенд: Ctrl+C
docker-compose down
```

---

## 5. Проверка работоспособности

| Сервис | URL |
|--------|-----|
| Frontend | http://localhost:5173 |
| Core API (docs) | http://localhost:8001/docs |
| Accessories API (docs) | http://localhost:8000/docs |

## 6. Опционально: локальные ML-модели

Только для альтернативного пути (uv), если используется локальная сегментация SAM2 + GroundingDINO:

```bash
mkdir -p core_service/models
wget -P core_service/models https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget -P core_service/models https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P core_service/models https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

Дополнить `core_service/.env`:

```env
SAM2_CHECKPOINT=./models/sam2.1_hiera_base_plus.pt
SAM2_CONFIG=sam2_hiera_b+
GROUNDING_DINO_CONFIG=./models/GroundingDINO_SwinT_OGC.py
GROUNDING_DINO_CHECKPOINT=./models/groundingdino_swint_ogc.pth
```
