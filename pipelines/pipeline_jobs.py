import os
import re
from typing import Dict, List, Optional

import requests
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText, MatchValue


JOB_COLLECTION = "job_instructions"


class Pipe:
    id = "job-instructions-rag"
    name = "Job Instructions RAG"

    class Valves(BaseModel):
        QDRANT_URL: str = Field(default=os.getenv("JOB_QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333")))
        OLLAMA_URL: str = Field(default=os.getenv("JOB_OLLAMA_URL", os.getenv("OLLAMA_URL", "http://localhost:11434")))
        EMBEDDING_MODEL: str = Field(default=os.getenv("JOB_EMBEDDING_MODEL", "bge-m3"))
        LLM_MODEL: str = Field(default=os.getenv("JOB_LLM_MODEL", "qwen3:8b"))
        COLLECTION: str = Field(default=os.getenv("JOB_COLLECTION", JOB_COLLECTION))
        TOP_K: int = Field(default=int(os.getenv("JOB_TOP_K", "8")))
        SHOW_CHUNKS: bool = Field(default=os.getenv("SHOW_CHUNKS", "false").lower() == "true")

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

    def _known_values(self, field: str, limit: int = 1000):
        qdrant = self._client()
        values = set()
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=self.valves.COLLECTION,
                limit=256,
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

    def _detect_value(self, question: str, field: str) -> Optional[str]:
        lowered = question.lower()
        try:
            for value in sorted(self._known_values(field), key=len, reverse=True):
                if value and value.lower() in lowered:
                    return value
        except Exception:
            return None
        return None

    def _keywords(self, question: str) -> List[str]:
        stop_words = {
            "что",
            "где",
            "как",
            "когда",
            "какие",
            "какой",
            "какая",
            "должен",
            "должна",
            "должны",
            "инструкция",
            "должностная",
        }
        keywords = []
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{4,}", question.lower()):
            if token not in stop_words and not re.fullmatch(r"\d+", token):
                keywords.append(token)
        return keywords[:4]

    def _build_filter(self, question: str) -> Optional[Filter]:
        conditions = []
        keyword_conditions = []
        role = self._detect_value(question, "role")
        section = self._detect_value(question, "section")
        clause = re.search(r"\b\d+(?:\.\d+)+\b", question)

        if role:
            conditions.append(FieldCondition(key="role", match=MatchValue(value=role)))
        if section:
            conditions.append(FieldCondition(key="section", match=MatchValue(value=section)))
        if clause:
            conditions.append(FieldCondition(key="clause_number", match=MatchValue(value=clause.group(0))))
        for keyword in self._keywords(question):
            keyword_conditions.append(FieldCondition(key="topic", match=MatchText(text=keyword)))
            keyword_conditions.append(FieldCondition(key="text", match=MatchText(text=keyword)))

        if not conditions and not keyword_conditions:
            return None
        return Filter(must=conditions or None, should=keyword_conditions or None)

    def _search(self, vector: List[float], query_filter: Optional[Filter]):
        qdrant = self._client()
        try:
            return qdrant.search(
                collection_name=self.valves.COLLECTION,
                query_vector=vector,
                query_filter=query_filter,
                limit=self.valves.TOP_K,
                with_payload=True,
            )
        except AttributeError:
            result = qdrant.query_points(
                collection_name=self.valves.COLLECTION,
                query=vector,
                query_filter=query_filter,
                limit=self.valves.TOP_K,
                with_payload=True,
            )
            return result.points
        except Exception:
            if query_filter is not None:
                return []
            raise

    def _ask_llm(self, question: str, context: str) -> str:
        system_prompt = (
            "Ты отвечаешь только по найденным должностным инструкциям.\n"
            'Если ответа в контексте нет, скажи: "В должностных инструкциях этого не найдено".\n'
            "Не используй общие знания.\n"
            "Не выдумывай.\n"
            "В ответе указывай должность, раздел, пункт и источник, если они есть.\n"
            "Отвечай на языке пользователя."
        )
        user_prompt = f"Контекст из должностных инструкций:\n\n{context}\n\nВопрос: {question}"
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
            timeout=240,
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

    def _context(self, hits) -> str:
        parts = []
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            parts.append(
                f"[{index}] Должность: {payload.get('role', '')}; "
                f"Документ: {payload.get('document_type', '')}; "
                f"Раздел: {payload.get('section', '')}; "
                f"Подраздел: {payload.get('subsection', '')}; "
                f"Пункт: {payload.get('clause_number', '')}; "
                f"Тема: {payload.get('topic', '')}; "
                f"Источник: {payload.get('source_file', '')}; "
                f"Score: {getattr(hit, 'score', 0):.3f}\n"
                f"{payload.get('text', '')}"
            )
        return "\n\n---\n\n".join(parts)

    def _sources_table(self, hits) -> str:
        rows = ["### Источники", "", "| Score | Должность | Раздел | Пункт | Тема | Источник |", "|---:|---|---|---|---|---|"]
        for hit in hits:
            payload = hit.payload or {}
            rows.append(
                f"| {getattr(hit, 'score', 0):.3f} | "
                f"{payload.get('role', '')} | "
                f"{payload.get('section', '')} | "
                f"{payload.get('clause_number', '')} | "
                f"{payload.get('topic', '')} | "
                f"{payload.get('source_file', '')} |"
            )
        return "\n".join(rows)

    def _debug_chunks(self, hits) -> str:
        if not self.valves.SHOW_CHUNKS:
            return ""
        parts = ["", "### Найденные чанки"]
        for index, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            text = (payload.get("text", "") or "")[:800]
            parts.append(
                f"\n#### Chunk {index}\n"
                f"score={getattr(hit, 'score', 0):.3f}\n\n"
                f"Должность: {payload.get('role', '')}\n"
                f"Раздел: {payload.get('section', '')}\n"
                f"Пункт: {payload.get('clause_number', '')}\n"
                f"Источник: {payload.get('source_file', '')}\n\n"
                f"{text}"
            )
        return "\n".join(parts)

    def pipe(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __task__=None,
        user_message=None,
        **kwargs,
    ):
        question = self._last_user_question(body)
        if not question and user_message:
            question = str(user_message)
        if not question:
            return "Не нашел вопрос пользователя в сообщениях."

        query_vector = self._ollama_embedding(question)
        query_filter = self._build_filter(question)
        hits = self._search(query_vector, query_filter)
        if query_filter and not hits:
            hits = self._search(query_vector, None)
        if not hits:
            return "В должностных инструкциях этого не найдено"

        answer = self._ask_llm(question, self._context(hits))
        return f"{answer}\n\n{self._sources_table(hits)}{self._debug_chunks(hits)}"


class Pipeline(Pipe):
    pass
