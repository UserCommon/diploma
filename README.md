# О приложении

Веб-приложение для визуального тюнинга автомобилей с помощью ИИ. Пользователь загружает фото машины, описывает желаемые изменения («поставь карбоновый спойлер», «смени колёса»), а ИИ-агент самостоятельно ищет деталь в базе, сегментирует нужную область и генерирует фотореалистичный результат.

## Архитектура

```
diploma_fixed/
├── accessories_processing/   # Сервис библиотеки деталей (FastAPI + SAQ)
├── core_service/             # ИИ-агент и пайплайн генерации (FastAPI + SAQ)
├── frontend/                 # SvelteKit UI
├── infra/                    # SQL-инициализация БД
└── docker-compose.yml        # postgres + redis + accessories сервисы
```

### Сервисы

| Сервис | Порт | Описание |
|--------|------|----------|
| `accessories_processing` | 8000 | Загрузка деталей, удаление фона, генерация эмбеддингов, семантический поиск |
| `core_service` | 8001 | LangChain ReAct агент: сегментация → поиск → инпейнтинг |
| `frontend` | 5173 | Studio — основной интерфейс пользователя |

## Быстрый старт

### 1. Инфраструктура (postgres + redis + accessories + frontend)

```bash
docker-compose up -d
```

### Локально без контейнеров
Чтобы развернуть без контейнеров смотрите `INSTRUCTIONS.md`

Открой `http://localhost:5173`

## Переменные окружения (core_service/.env)

Для POLZA_AI_API_KEY, прилагаю ключ на 200 рублей генераций: pza_c9TngHDFkhuvtE-fme4dML1RcqM28Bm4

```env
# Инфраструктура
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/core_db
REDIS_URL=redis://localhost:6379
STORAGE_PATH=./storage
CORS_ORIGINS=http://localhost:5173
ACCESSORIES_SERVICE_URL=http://localhost:8000

# LLM агент
LLM_PROVIDER=openai_compatible   # openai | anthropic | openai_compatible
LLM_API_KEY=
LLM_MODEL=moonshotai/kimi-k2.6
LLM_BASE_URL=https://polza.ai/api/v1   # только для openai_compatible

# Провайдер инпейнтинга
INPAINT_PROVIDER=polza   # polza | genapi | klein | kontext | gpt_image | sdxl | diffusers

# polza.ai (FLUX.2 Pro — основной)
POLZA_AI_API_KEY=
POLZA_MODEL=black-forest-labs/flux.2-pro

# gen-api.ru (FLUX Inpainting / Klein / Kontext)
GENAPI_API_KEY=
GENAPI_MODEL=inpainting
GENAPI_KLEIN_MODEL=4B Standard
GENAPI_KONTEXT_MODEL=max

# OpenAI GPT Image
GPT_IMAGE_MODEL=openai/gpt-image-1.5

# Локальные модели (опционально)
SAM2_CHECKPOINT=./models/sam2_hiera_base_plus.pt
SAM2_CONFIG=sam2_hiera_b+
GROUNDING_DINO_CONFIG=./models/GroundingDINO_SwinT_OGC.py
GROUNDING_DINO_CHECKPOINT=./models/groundingdino_swint_ogc.pth
HF_TOKEN=   # для загрузки FLUX Fill с HuggingFace
```

## ML-модели (сегментация)

Для сегментации нужны SAM2 и GroundingDINO. Скачать чекпойнты:

```bash
mkdir -p core_service/models
# SAM2
wget -P core_service/models https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

# GroundingDINO
wget -P core_service/models https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P core_service/models https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py
```

## API

### Accessories Service (`localhost:8000`)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/parts/upload` | Загрузить деталь (фото → удаление фона → эмбеддинг) |
| `GET` | `/api/parts/` | Список деталей |
| `POST` | `/api/parts/search` | Семантический поиск по описанию |
| `DELETE` | `/api/parts/{id}` | Удалить деталь |

### Core Service (`localhost:8001`)

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/agent/sessions` | Создать сессию (фото авто + промпт) |
| `GET` | `/api/agent/sessions/{id}` | Статус и результаты сессии |
| `GET` | `/api/agent/sessions/{id}/stream` | SSE-стрим событий агента |
| `GET` | `/storage/{category}/{filename}` | Отдача сохранённых изображений |

## Провайдеры инпейнтинга

| Провайдер | Ключ `.env` | Маска | Референс | Примечание |
|-----------|-------------|-------|----------|------------|
| polza.ai FLUX.2 Pro | `polza` | ✅ | ✅ | Лучший результат |
| gen-api FLUX Inpaint | `genapi` | ✅ | ✅ (pre-paste) | Маска-based |
| gen-api Flux 2 Klein | `klein` | — | ✅ | img2img |
| gen-api Flux Kontext | `kontext` | — | ✅ | img2img |
| GPT Image 1 | `gpt_image` | soft | — | Регенерирует всё |
| FLUX Fill (локально) | `diffusers` | ✅ | — | Требует GPU |
| SDXL (локально) | `sdxl` | ✅ | — | Требует GPU |

## Стек

- **Backend**: FastAPI, SQLAlchemy (async), Alembic, SAQ
- **ML**: LangChain ReAct agent, SAM2, GroundingDINO, sentence-transformers
- **БД**: PostgreSQL 16 + pgvector
- **Очередь**: Redis + SAQ
- **Frontend**: SvelteKit, Vite
- **Инфра**: Docker Compose
