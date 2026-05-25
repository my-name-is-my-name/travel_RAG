# Travel RAG

Локальный RAG по travel-заметкам, где код живет отдельно от Obsidian Vault.

В примерах ниже используются такие условные пути:

- `PROJECT_ROOT=/absolute/path/to/travel-rag`
- `ROOT_DATA_DIR=/absolute/path/to/vacations`

## Где что лежит

- Код и запуск: `PROJECT_ROOT`
- Заметки и данные: `ROOT_DATA_DIR`
- Исходные заметки для sync: `raw_notes/`
- Готовые structured-заметки для индекса: `structured_notes/`

Важно: в этом repo нет локальных `raw_notes/` и `structured_notes`. Они читаются через `ROOT_DATA_DIR`.

## Текущая конфигурация

`.env` использует:

- `ROOT_DATA_DIR=/absolute/path/to/vacations`
- answer generation: `LLM_PROVIDER=openai`, `LLM_MODEL=deepseek-v4-pro`
- formatter: `FORMATTER_PROVIDER=openai`, `FORMATTER_MODEL=deepseek-v4-flash`
- embeddings: `bge-m3`
- Qdrant collection: `travel_notes`

## Первый запуск

```bash
cd /absolute/path/to/travel-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Поднять сервисы

### Qdrant

```bash
cd /absolute/path/to/travel-rag
docker compose -f docker-compose.qdrant.yml up -d
```

Проверка:

```bash
curl http://localhost:6333
```

### Ollama

Нужен для embeddings `bge-m3`.

Запусти приложение Ollama и проверь:

```bash
curl http://localhost:11434/api/tags
```

Если модели нет:

```bash
ollama pull bge-m3
```

### Pipelines backend

```bash
cd /absolute/path/to/travel-rag
docker compose -f docker-compose.pipelines.yml up -d
```

Проверка:

```bash
curl http://localhost:9099
```

Ожидаемый ответ:

```json
{"status":true}
```

## Как пользоваться

### 1. Добавил новую raw-заметку

Клади новый markdown-файл сюда:

```text
ROOT_DATA_DIR/raw_notes/
```

### 2. Сформировать structured-файл и проиндексировать его

```bash
cd /absolute/path/to/travel-rag
source .venv/bin/activate
python sync.py --file "raw_notes/NAME.md"
```

Это делает:

1. читает `raw_notes/NAME.md`
2. вызывает formatter `deepseek-v4-flash`
3. сохраняет результат в `structured_notes/NAME.md`
4. удаляет старые чанки этого `source_file` из Qdrant
5. заново индексирует заметку

### 3. Переиндексировать все готовые structured-заметки

Это самая полезная команда, если поиск ведет себя странно или ты поднял новый Qdrant:

```bash
cd /absolute/path/to/travel-rag
source .venv/bin/activate
python sync.py --skip-format --force
```

### 4. Переиндексировать один structured-файл без formatter

```bash
python sync.py --skip-format --file "structured_notes/2024 - Стамбул 23-30.11.24.md"
```

## Если RAG отвечает странно

Проверь по порядку:

1. поднят ли Qdrant
2. поднят ли pipelines backend
3. отвечает ли Ollama
4. есть ли нужные поездки в `structured_notes/`
5. была ли выполнена переиндексация:

```bash
python sync.py --skip-format --force
```

Типичный симптом: если в индексе оказалась только одна поездка, RAG начинает тянуть ее в источники почти на любой вопрос.

## Ошибки formatter

Если formatter вернул невалидный Markdown или запрос сорвался, ошибка пишется сюда:

```text
PROJECT_ROOT/errors/
```

## Полезные команды

### Статус Qdrant

```bash
curl http://localhost:6333/collections/travel_notes
```

### Статус pipelines

```bash
curl http://localhost:9099
```

### Логи pipelines

```bash
docker logs travel-rag-pipelines-1
```

### Какие контейнеры живы

```bash
docker ps
```

## OpenWebUI

`Travel RAG` backend сам по себе живет на `9099`. Если нужен интерфейс OpenWebUI, его надо запускать отдельно.

## Что не трогать руками

- не редактируй `structured_notes` как основной источник правды, если потом хочешь честно пересобирать их из raw
- не меняй имена файлов в `raw_notes` и `structured_notes` несогласованно: `source_file` завязан на имя raw-файла
- не путай старые `vacations-*` контейнеры с новыми `travel-rag-*`
