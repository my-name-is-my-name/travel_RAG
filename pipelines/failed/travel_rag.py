import os
import re
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


EVENT_TYPE_ALIASES = {
    "музей": "музей",
    "музеи": "музей",
    "выставка": "выставка",
    "выставку": "выставка",
    "театр": "театр",
    "спектакль": "театр",
    "еда": "еда",
    "ели": "еда",
    "ел": "еда",
    "ресторан": "еда",
    "ресторане": "еда",
    "кафе": "еда",
    "обед": "еда",
    "ужин": "еда",
    "завтрак": "еда",
    "пляж": "пляж",
    "пляжные": "пляж",
    "купались": "пляж",
    "бег": "спорт",
    "спорт": "спорт",
    "сап": "спорт",
    "sup": "спорт",
    "транспорт": "транспорт",
    "поезд": "транспорт",
    "самолет": "транспорт",
    "такси": "транспорт",
    "жилье": "жилье",
    "отель": "жилье",
    "апартаменты": "жилье",
}


class Pipe:
    id = "travel-rag"
    name = "Travel RAG"

    class Valves(BaseModel):
        QDRANT_URL: str = Field(default=os.getenv("QDRANT_URL", "http://localhost:6333"))
        OLLAMA_URL: str = Field(default=os.getenv("OLLAMA_URL", "http://localhost:11434"))
        EMBEDDING_MODEL: str = Field(default=os.getenv("EMBEDDING_MODEL", "bge-m3"))
        LLM_MODEL: str = Field(default=os.getenv("LLM_MODEL", "gemma3:4b"))
        COLLECTION: str = Field(default=os.getenv("COLLECTION", "travel_notes"))
        TOP_K: int = Field(default=int(os.getenv("TOP_K", "8")))

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": self.id, "name": self.name}]

    def _client(self):
        return QdrantClient(url=self.valves.QDRANT_URL)

    def _ollama_embedding(self, text: str) -> List[float]:
        base_url = self.valves.OLLAMA_URL.rstrip("/")
        last_error = None
        for path, payload in (
            ("/api/embed", {"model": self.valves.EMBEDDING_MODEL, "input": text}),
            ("/api/embeddings", {"model": self.valves.EMBEDDING_MODEL, "prompt": text}),
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
            f"Ollama embeddings недоступны: {last_error}. Проверьте {base_url} и модель {self.valves.EMBEDDING_MODEL}."
        )

    def _known_payload_values(self, field: str, limit: int = 1000):
        client = self._client()
        values = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.valves.COLLECTION,
                limit=min(limit, 256),
                offset=offset,
                with_payload=[field],
                with_vectors=False,
            )
            for point in points:
                value = (point.payload or {}).get(field)
                if value:
                    values.add(str(value))
            if offset is None or len(values) >= limit:
                break
        return values

    def _detect_city(self, question: str) -> Optional[str]:
        lowered = question.lower()
        try:
            for city in sorted(self._known_payload_values("city"), key=len, reverse=True):
                if city and city.lower() in lowered:
                    return city
        except Exception:
            return None
        return None

    def _detect_type(self, question: str) -> Optional[str]:
        lowered = question.lower()
        for word, event_type in EVENT_TYPE_ALIASES.items():
            if re.search(rf"(^|\W){re.escape(word)}($|\W)", lowered):
                return event_type
        return None

    def _build_filter(self, question: str) -> Optional[Filter]:
        conditions = []
        city = self._detect_city(question)
        event_type = self._detect_type(question)
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
        year_match = re.search(r"\b(20\d{2})\b", question)

        if city:
            conditions.append(FieldCondition(key="city", match=MatchValue(value=city)))
        if date_match:
            conditions.append(FieldCondition(key="date", match=MatchValue(value=date_match.group(0))))
        elif year_match:
            conditions.append(FieldCondition(key="year", match=MatchValue(value=year_match.group(1))))
        if event_type:
            conditions.append(FieldCondition(key="type", match=MatchValue(value=event_type)))

        if not conditions:
            return None
        return Filter(must=conditions)

    def _search(self, vector: List[float], query_filter: Optional[Filter]):
        client = self._client()
        try:
            return client.search(
                collection_name=self.valves.COLLECTION,
                query_vector=vector,
                query_filter=query_filter,
                limit=self.valves.TOP_K,
                with_payload=True,
            )
        except AttributeError:
            result = client.query_points(
                collection_name=self.valves.COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=self.valves.TOP_K,
                with_payload=True,
            )
            return result.points

    def _ask_llm(self, question: str, context: str) -> str:
        system_prompt = (
            "Ты отвечаешь только по найденным заметкам пользователя.\n"
            'Если ответа в контексте нет, скажи: "В заметках этого не найдено".\n'
            "Не используй общие знания.\n"
            "Не выдумывай.\n"
            "Указывай дату, город и источник, если они есть.\n"
            "Отвечай на языке пользователя."
        )
        user_prompt = f"Контекст из заметок:\n\n{context}\n\nВопрос: {question}"
        response = requests.post(
            f"{self.valves.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": self.valves.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
        return (response.json().get("message") or {}).get("content", "").strip()

    def _last_user_question(self, body) -> str:
        messages = body.get("messages", []) if isinstance(body, dict) else []
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(
                        part.get("text", "") for part in content if isinstance(part, dict)
                    ).strip()
        return ""

    def pipe(self, body: dict, __user__=None, __event_emitter__=None, __task__=None):
        question = self._last_user_question(body)
        if not question:
            return "Не нашел вопрос пользователя в сообщениях."

        query_vector = self._ollama_embedding(question)
        query_filter = self._build_filter(question)
        hits = self._search(query_vector, query_filter)
        if query_filter and not hits:
            hits = self._search(query_vector, None)

        if not hits:
            return "В заметках этого не найдено."

        context_parts = []
        sources = []
        seen_sources = set()
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            context_parts.append(
                f"[{index}] Источник: {payload.get('source_file', '')}; "
                f"Дата: {payload.get('date', '')}; "
                f"Город: {payload.get('city', '')}; "
                f"Тип: {payload.get('type', '')}; "
                f"Событие: {payload.get('event_title', '')}\n"
                f"{payload.get('text', '')}"
            )
            source_key = (
                payload.get("source_file", ""),
                payload.get("date", ""),
                payload.get("city", ""),
                payload.get("event_title", ""),
            )
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(source_key)

        answer = self._ask_llm(question, "\n\n---\n\n".join(context_parts))
        source_lines = [
            f"- {source_file}, {date}, {city}, {event_title}".strip()
            for source_file, date, city, event_title in sources[: self.valves.TOP_K]
        ]
        return answer + "\n\nИсточники:\n" + "\n".join(source_lines)


class Pipeline(Pipe):
    pass
