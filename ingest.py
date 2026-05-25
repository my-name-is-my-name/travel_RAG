import argparse
import hashlib
import os
import re
import uuid
from pathlib import Path

import frontmatter
import requests
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from path_config import CODE_ROOT, DATA_ROOT

STRUCTURED_DIR = DATA_ROOT / "structured_notes"
DEFAULTS = {
    "QDRANT_URL": "http://localhost:6333",
    "OLLAMA_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "bge-m3",
    "COLLECTION": "travel_notes",
}
EVENT_TYPE_VALUES = {
    "транспорт",
    "жилье",
    "еда",
    "прогулка",
    "музей",
    "выставка",
    "театр",
    "природа",
    "пляж",
    "спорт",
    "покупка",
    "работа",
    "настроение",
    "итог дня",
    "другое",
}


def load_config():
    load_dotenv(CODE_ROOT / ".env")
    config = DEFAULTS.copy()
    config_path = CODE_ROOT / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in data.items():
            config[str(key).upper()] = str(value)
    for key in list(config):
        config[key] = os.getenv(key, config[key])
    return config


def ollama_embedding(text, config):
    base_url = config["OLLAMA_URL"].rstrip("/")
    model = config["EMBEDDING_MODEL"]
    last_error = None
    for path, payload in (
        ("/api/embed", {"model": model, "input": text}),
        ("/api/embeddings", {"model": model, "prompt": text}),
    ):
        try:
            response = requests.post(f"{base_url}{path}", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            if "embedding" in data:
                return data["embedding"]
            if data.get("embeddings"):
                return data["embeddings"][0]
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(
        f"Ollama embeddings недоступны: {last_error}. Проверьте {base_url} и модель {model}."
    )


def ensure_collection(client, collection_name, vector_size):
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name in existing:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def delete_source(client, collection_name, source_file):
    existing = {collection.name for collection in client.get_collections().collections}
    if collection_name not in existing:
        return
    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            )
        ),
    )


def parse_context_type(context):
    lowered = context.lower()
    match = re.search(r"тип\s*:\s*([^;,\n]+)", lowered)
    if match:
        value = match.group(1).strip()
        if value in EVENT_TYPE_VALUES:
            return value
    for event_type in EVENT_TYPE_VALUES:
        if event_type in lowered:
            return event_type
    return "другое"


def parse_events(path):
    post = frontmatter.load(path)
    metadata = dict(post.metadata)
    source_file = metadata.get("source_file") or path.name
    trip = metadata.get("trip") or path.stem
    companions = metadata.get("companions") or []
    lines = post.content.splitlines()

    current_date = ""
    current_city = ""
    current_title = None
    current_lines = []
    events = []

    def flush_event():
        if not current_title:
            return
        raw = "\n".join(current_lines).strip()
        if not raw:
            return
        context = ""
        body_lines = []
        for line in raw.splitlines():
            if line.strip().lower().startswith("контекст:"):
                context = line.strip()
            else:
                body_lines.append(line)
        text = "\n".join(body_lines).strip()
        events.append(
            {
                "trip": trip,
                "companions": companions,
                "date": current_date,
                "city": current_city,
                "type": parse_context_type(context),
                "source_file": source_file,
                "event_title": current_title.strip(),
                "context": context,
                "text": text or raw,
            }
        )

    for line in lines:
        day_match = re.match(r"^##\s+(\d{4}-\d{2}-\d{2}|не уверено)\s+[—-]\s+(.+?)\s*$", line)
        if day_match:
            flush_event()
            current_title = None
            current_lines = []
            current_date = day_match.group(1).strip()
            current_city = day_match.group(2).strip()
            continue
        if re.match(r"^##\s+Факты для RAG\s*$", line, flags=re.IGNORECASE):
            flush_event()
            break
        title_match = re.match(r"^###\s+(.+?)\s*$", line)
        if title_match:
            flush_event()
            current_title = title_match.group(1).strip()
            current_lines = []
            continue
        if current_title:
            current_lines.append(line)

    flush_event()
    return events


def split_large_event(event, max_words=260):
    words = event["text"].split()
    if len(words) <= max_words:
        return [event["text"]]
    chunks = []
    for start in range(0, len(words), max_words):
        part = " ".join(words[start : start + max_words])
        prefix = (
            f"Дата: {event['date']}\n"
            f"Город: {event['city']}\n"
            f"Поездка: {event['trip']}\n"
            f"Тип: {event['type']}\n"
            f"Событие: {event['event_title']}\n\n"
        )
        chunks.append(prefix + part)
    return chunks


def point_uuid(source_file, date, event_title, chunk_index):
    raw_id = f"{source_file}|{date}|{event_title}|{chunk_index}"
    return str(uuid.UUID(hashlib.md5(raw_id.encode("utf-8")).hexdigest()))


def build_embedding_text(event, text):
    return (
        f"Поездка: {event['trip']}\n"
        f"Компаньоны: {', '.join(event.get('companions') or [])}\n"
        f"Дата: {event['date']}\n"
        f"Город: {event['city']}\n"
        f"Тип: {event['type']}\n"
        f"Событие: {event['event_title']}\n"
        f"Источник: {event['source_file']}\n\n"
        f"{text}"
    )


def event_year(date):
    match = re.match(r"^(\d{4})", date or "")
    return match.group(1) if match else ""


def ingest_file(path, config, force=False):
    if not path.exists():
        raise FileNotFoundError(f"Structured-файл не найден: {path}")

    events = parse_events(path)
    source_file = frontmatter.load(path).metadata.get("source_file") or path.name
    client = QdrantClient(url=config["QDRANT_URL"])
    collection_name = config["COLLECTION"]
    points = []
    vector_size = None

    if force:
        delete_source(client, collection_name, source_file)

    for event in events:
        for chunk_index, chunk_text in enumerate(split_large_event(event)):
            embedding_text = build_embedding_text(event, chunk_text)
            vector = ollama_embedding(embedding_text, config)
            if vector_size is None:
                vector_size = len(vector)
                ensure_collection(client, collection_name, vector_size)
                if not force:
                    delete_source(client, collection_name, source_file)
            point_id = point_uuid(event["source_file"], event["date"], event["event_title"], chunk_index)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "id": point_id,
                        "trip": event["trip"],
                        "companions": event.get("companions") or [],
                        "date": event["date"],
                        "year": event_year(event["date"]),
                        "city": event["city"],
                        "type": event["type"],
                        "source_file": event["source_file"],
                        "event_title": event["event_title"],
                        "text": embedding_text,
                    },
                )
            )

    if points:
        client.upsert(collection_name=collection_name, points=points)
    return len(points)


def iter_structured_files(single_file=None):
    if single_file:
        path = Path(single_file)
        if not path.is_absolute():
            path = ROOT / path
        return [path]
    return sorted(STRUCTURED_DIR.glob("*.md"))


def main():
    parser = argparse.ArgumentParser(description="Index structured travel notes into Qdrant.")
    parser.add_argument("--file", help="Path to one structured markdown file.")
    parser.add_argument("--force", action="store_true", help="Delete old chunks before reindexing.")
    args = parser.parse_args()

    config = load_config()
    STRUCTURED_DIR.mkdir(exist_ok=True)

    total = 0
    for path in iter_structured_files(args.file):
        count = ingest_file(path, config, force=True if args.force else False)
        total += count
        print(f"Indexed {count} chunks: {path}")
    print(f"Done. Total chunks: {total}")


if __name__ == "__main__":
    main()
