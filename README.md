# LLM-агент для правовых справок

RAG-система с FastAPI и Telegram-ботом: гибридный retrieval (FTS + pgvector), reranker, LLM-агент нормализации запроса, верификатор ответа и единый evaluation workflow (автооценка + human review).

## Текущая архитектура

### Основной поток запроса

1. Клиент (API `/query` или Telegram) отправляет вопрос.
2. `ConversationOrchestrator` запускает пайплайн:
   - safety gate;
   - (опционально) `LegalQueryAgent` для нормализации запроса;
   - retrieval (`llamaindex_hybrid` или `explicit_hybrid`);
   - (опционально) reranker;
   - генерация ответа LLM + цитаты;
   - (опционально) `AnswerVerifier`.
3. `RAGService` возвращает ответ и логирует взаимодействие в `interaction_logs`.

### Структура проекта

- `src/main.py` — точка входа FastAPI, lifecycle БД.
- `src/api/` — роуты API (`/query`, `/health`, `/admin/ingest`) и auth.
- `src/bot/` — Telegram-бот (aiogram).
- `src/services/` — `RAGService`, логирование взаимодействий.
- `src/dialog/` — оркестратор диалога и pre-checks.
- `src/rag/` — retrieval, fusion (RRF), reranker, генерация, pipeline.
- `src/agent/` — LLM-агент нормализации запроса и верификатор ответа.
- `src/knowledge/` — ingestion, chunking, индексация в pgvector.
- `src/db/` — asyncpg pool и операции с `interaction_logs`.
- `evaluation/` — benchmark/eval код и отчеты.
- `scripts/` — CLI-скрипты запуска, индексации и оценки.
- `knowledge_base/` — исходные документы (ТК РФ, FAQ Роструда, uploads).

## Стек и модели

### Технологический стек

- Python 3.11+
- FastAPI + Uvicorn
- aiogram (Telegram)
- PostgreSQL + pgvector
- LlamaIndex
- asyncpg / psycopg2
- Hugging Face Hub / sentence-transformers
- RAGAS + datasets (evaluation)

### Используемые модели (дефолт из `config` / `.env.example`)

- **LLM (генерация, агент, верификатор):**
  - `LLM_PROVIDER=openrouter` (по умолчанию)
  - `LLM_MODEL=qwen/qwen3-next-80b-a3b-instruct`
  - Поддерживаемые провайдеры: `openrouter`, `hf`, `deepseek`
- **Embedding model (dense retrieval):**
  - `EMBEDDING_MODEL=intfloat/multilingual-e5-large`
- **Encoder для rerank (cross-encoder):**
  - `RERANKER_MODEL=qilowoq/bge-reranker-v2-m3-en-ru`

## Переменные окружения

Минимально необходимые:

- `DATABASE_URL`
- `DATABASE_URL_SYNC`
- `HUGGINGFACE_HUB_TOKEN`
- `OPENROUTER_API_KEY` (если `LLM_PROVIDER=openrouter`)
- `TELEGRAM_BOT_TOKEN` (если нужен бот)
- `REDIS_URL` (для cache/rate limit)

Базовая настройка:

```bash
cp .env.example .env
```

## Запуск через Docker

### 1) Поднять сервисы

```bash
docker compose up --build -d
```

Поднимаются:
- `db` — PostgreSQL с pgvector (`pgvector/pgvector:pg16`)
- `redis` — кеш и rate-limit
- `app` — FastAPI
- `bot` — Telegram-бот

### 2) Проверить health API

```bash
curl http://localhost:8000/health
```

### 3) Выполнить первичную индексацию базы знаний

```bash
docker compose exec app python -m scripts.index_knowledge --reset
```

После этого API готов к полноценным запросам.

Дополнительно в production-ready режиме включены:
- кэш ответов и retrieval в Redis;
- rate limit `100 req/min` на пользователя (или IP для анонимных запросов);
- retry для LLM-вызовов и fallback-ответ `Недостаточно данных` при ошибке.

## API

- `POST /query` — основной RAG-запрос.
- `GET /health` — проверка сервиса.
- `POST /admin/ingest` — загрузка и инкрементальная индексация `.md/.txt` (требует `X-API-KEY`).

Пример `admin/ingest`:

```bash
curl -X POST http://localhost:8000/admin/ingest \
  -H "X-API-KEY: your_admin_secret" \
  -F "file=@document.md" \
  -F "source_name=custom"
```

## Оценка на эталонных вопросах/ответах

Эталонный набор: `evaluation/benchmarks/reference_qa.jsonl`.

### Вариант A: локально (venv)

```bash
python -m scripts.eval_ragas
```

### Вариант B: в Docker-контуре

```bash
docker compose exec app python -m scripts.eval_ragas
```

### Что получится

В `evaluation/reports/` создаются артефакты:

- `<dataset>_<timestamp>.jsonl` — полный построчный отчет (`question`, `reference_answer`, `model_answer`, `context_texts`, мета).
- `<dataset>_<timestamp>.csv` — компактный табличный отчет.
- `<dataset>_<timestamp>_human_review.csv` — шаблон для экспертной оценки.
- `<dataset>_<timestamp>_ragas.json` — агрегированные RAGAS-метрики.

### Как подключить экспертную оценку

1. Передать эксперту файл `<run>_human_review.csv`.
2. После заполнения провалидировать:

```bash
python -m scripts.validate_human_review --input evaluation/reports/<run>_human_review.csv
```

3. Получить summary:
   - `<run>_human_review_summary.json`
   - `<run>_human_review_summary.csv`

Правила контрактов артефактов зафиксированы в коде `evaluation/core.py` и `scripts/validate_human_review.py`.

## Пакетная оценка FAQ Роструда

```bash
python -m scripts.batch_eval_rostrud --limit 20
python -m scripts.batch_eval_rostrud --limit 20 --ragas
```

В Docker:

```bash
docker compose exec app python -m scripts.batch_eval_rostrud --limit 20
docker compose exec app python -m scripts.batch_eval_rostrud --limit 20 --ragas
```
