import argparse
import os
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import frontmatter
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from path_config import DATA_ROOT


STRUCTURED_DIR = DATA_ROOT / "job_structured"

QDRANT_URL = os.getenv("JOB_QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333"))
OLLAMA_URL = os.getenv("JOB_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434"))
EMBEDDING_MODEL = os.getenv("JOB_EMBEDDING_MODEL", "bge-m3")
COLLECTION = os.getenv("JOB_COLLECTION", "job_instructions")


def embedding(text: str) -> List[float]:
    base_url = OLLAMA_URL.rstrip("/")
    last_error = None

    for path, payload in (
        ("/api/embed", {"model": EMBEDDING_MODEL, "input": text}),
        ("/api/embeddings", {"model": EMBEDDING_MODEL, "prompt": text}),
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
        f"Ollama embeddings недоступны: {last_error}. "
        f"Проверьте {base_url} и модель {EMBEDDING_MODEL}."
    )


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(qdrant: QdrantClient, vector_size: int) -> None:
    collections = {collection.name for collection in qdrant.get_collections().collections}

    if COLLECTION in collections:
        ensure_payload_indexes(qdrant)
        return

    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    ensure_payload_indexes(qdrant)


def ensure_payload_indexes(qdrant: QdrantClient) -> None:
    keyword_fields = [
        "role",
        "document_type",
        "department",
        "section",
        "subsection",
        "clause_number",
        "source_file",
    ]
    text_fields = ["topic", "text"]

    for field in keyword_fields:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            pass

    for field in text_fields:
        try:
            qdrant.create_payload_index(
                collection_name=COLLECTION,
                field_name=field,
                field_schema=PayloadSchemaType.TEXT,
                wait=True,
            )
        except Exception:
            pass


def delete_source(qdrant: QdrantClient, source_file: str) -> None:
    collections = {collection.name for collection in qdrant.get_collections().collections}

    if COLLECTION not in collections:
        return

    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=source_file),
                )
            ]
        ),
        wait=True,
    )


def deterministic_id(source_file: str, clause_number: str, topic: str, chunk_index: int) -> str:
    raw = f"{source_file}|{clause_number}|{topic}|{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def split_large_text(text: str, max_chars: int = 2600) -> List[str]:
    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    parts = []
    current = []
    current_len = 0

    for line in text.splitlines():
        add_len = len(line) + 1

        if current and current_len + add_len > max_chars:
            parts.append("\n".join(current).strip())
            current = []
            current_len = 0

        current.append(line)
        current_len += add_len

    if current:
        parts.append("\n".join(current).strip())

    return [part for part in parts if part]


def parse_context(lines: Iterable[str]) -> Dict[str, str]:
    context = {}

    for line in lines:
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        context[key.strip().lower()] = value.strip()

    return context


def parse_structured(path: Path) -> List[Dict[str, str]]:
    post = frontmatter.loads(path.read_text(encoding="utf-8"))

    metadata = post.metadata
    role = str(metadata.get("role", "")).strip()
    document_type = str(metadata.get("document_type", "Должностная инструкция")).strip()
    department = str(metadata.get("department", "")).strip()
    source_file = str(metadata.get("source_file", path.name)).strip()

    chunks = []
    section = ""

    blocks = re.split(r"\n(?=### Пункт )", post.content)

    for block in blocks:
        section_match = re.findall(r"^##\s+(.+)$", block, flags=re.MULTILINE)
        if section_match:
            section = section_match[-1].strip()

        title_match = re.search(
            r"^### Пункт\s+(.+?)\s+—\s+(.+)$",
            block,
            flags=re.MULTILINE,
        )
        if not title_match:
            continue

        clause_number = title_match.group(1).strip()
        topic = title_match.group(2).strip()

        after_title = block[title_match.end() :].strip()
        if "Контекст:" not in after_title:
            continue

        _, after_context = after_title.split("Контекст:", 1)

        context_lines = []
        text_lines = []
        in_text = False

        for line in after_context.splitlines():
            if not in_text and not line.strip():
                in_text = True
                continue

            if in_text:
                text_lines.append(line)
            else:
                context_lines.append(line)

        context = parse_context(context_lines)
        text = "\n".join(text_lines).strip()

        if not text:
            continue

        chunks.append(
            {
                "role": context.get("должность", role),
                "document_type": context.get("документ", document_type),
                "department": department,
                "section": context.get("раздел", section),
                "subsection": context.get("подраздел", ""),
                "clause_number": context.get("пункт", clause_number),
                "topic": context.get("тема", topic),
                "source_file": source_file,
                "text": text,
            }
        )

    return chunks


def embedding_text(payload: Dict[str, str], text: str) -> str:
    return (
        f"Должность: {payload['role']}\n"
        f"Документ: {payload['document_type']}\n"
        f"Раздел: {payload['section']}\n"
        f"Подраздел: {payload['subsection']}\n"
        f"Пункт: {payload['clause_number']}\n"
        f"Тема: {payload['topic']}\n"
        f"Источник: {payload['source_file']}\n\n"
        f"{text}"
    )


def ingest_file(path: Path, force: bool = False) -> int:
    qdrant = client()
    parsed = parse_structured(path)

    if not parsed:
        return 0

    source_file = parsed[0]["source_file"]
    delete_source(qdrant, source_file)

    points = []
    vector_size: Optional[int] = None

    for chunk in parsed:
        parts = split_large_text(chunk["text"])

        for chunk_index, part in enumerate(parts):
            payload = dict(chunk)
            payload["text"] = part

            if len(parts) > 1:
                payload["topic"] = f"{payload['topic']} — часть {chunk_index + 1}"

            vector = embedding(embedding_text(payload, part))

            if vector_size is None:
                vector_size = len(vector)
                ensure_collection(qdrant, vector_size)

            points.append(
                PointStruct(
                    id=deterministic_id(
                        payload["source_file"],
                        payload["clause_number"],
                        payload["topic"],
                        chunk_index,
                    ),
                    vector=vector,
                    payload=payload,
                )
            )

    if points:
        qdrant.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

    return len(points)


def iter_files(file_arg: Optional[str]) -> Iterable[Path]:
    if file_arg:
        path = Path(file_arg)

        if not path.is_absolute():
            path = ROOT / path

        yield path
        return

    yield from sorted(STRUCTURED_DIR.glob("*.md"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index job instruction structured markdown into Qdrant."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete old points by source_file before indexing.",
    )
    parser.add_argument(
        "--file",
        help="Index one file from job_structured/.",
    )

    args = parser.parse_args()

    total = 0

    for path in iter_files(args.file):
        count = ingest_file(path, force=args.force)
        total += count
        print(f"Indexed {count} chunks: {path}")

    print(f"Done. Total chunks: {total}")


if __name__ == "__main__":
    main()
