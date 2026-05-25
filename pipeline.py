import json
import os
import re
from typing import List, Optional

import requests
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


QUERY_STOP_WORDS = {
    "где",
    "что",
    "как",
    "когда",
    "какие",
    "какой",
    "какая",
    "какое",
    "мы",
    "нас",
    "нам",
    "был",
    "была",
    "были",
    "делали",
}

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
    "сапы": "спорт",
    "сапах": "спорт",
    "сапами": "спорт",
    "сапов": "спорт",
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
        LLM_PROVIDER: str = Field(default=os.getenv("LLM_PROVIDER", "ollama"))
        OPENAI_BASE_URL: str = Field(default=os.getenv("OPENAI_BASE_URL", ""))
        OPENAI_API_KEY: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
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

    def _stem_token(self, token: str) -> str:
        token = token.lower().replace("ё", "е")
        for ending in (
            "иями",
            "ями",
            "ами",
            "ого",
            "ему",
            "ыми",
            "ими",
            "ах",
            "ях",
            "ов",
            "ев",
            "ом",
            "ем",
            "ой",
            "ый",
            "ий",
            "ая",
            "ое",
            "ые",
            "ие",
            "ам",
            "ям",
            "а",
            "я",
            "ы",
            "и",
            "е",
            "у",
            "ю",
        ):
            if len(token) > len(ending) + 2 and token.endswith(ending):
                return token[: -len(ending)]
        return token

    def _query_stems(self, question: str):
        stems = []
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", question.lower()):
            if token in QUERY_STOP_WORDS or re.fullmatch(r"\d+", token):
                continue
            stem = self._stem_token(token)
            if len(stem) >= 3:
                stems.append(stem)
        return set(stems)

    def _lexically_relevant_hits(self, question: str, hits):
        stems = self._query_stems(question)
        if not stems:
            return hits

        relevant = []
        for hit in hits:
            payload = hit.payload or {}
            haystack = self._stem_token(
                f"{payload.get('event_title', '')} {payload.get('text', '')}".lower()
            )
            haystack_stems = {
                self._stem_token(token)
                for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]{3,}", haystack)
            }
            if stems & haystack_stems:
                relevant.append(hit)
        return relevant or hits

    def _llm_messages(self, question: str, context: str):
        system_prompt = (
            "Ты отвечаешь только по найденным заметкам пользователя.\n"
            'Если ответа в контексте нет, скажи: "В заметках этого не найдено".\n'
            "Если в контексте есть события, явно связанные с вопросом, обязательно ответь по ним.\n"
            "Для вопросов 'где' сначала перечисляй города и конкретные места/события из контекста.\n"
            "Не используй общие знания.\n"
            "Не выдумывай.\n"
            "Указывай дату, город и источник, если они есть.\n"
            "Отвечай на языке пользователя."
        )
        user_prompt = (
            f"Контекст из заметок:\n\n{context}\n\n"
            f"Вопрос: {question}\n\n"
            "Ответь только по этому контексту. Если релевантные события найдены в контексте, "
            "суммируй их, а не отвечай, что ничего не найдено."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _ask_llm(self, question: str, context: str) -> str:
        messages = self._llm_messages(question, context)
        if self.valves.LLM_PROVIDER.lower() in {"openai", "openai-compatible", "openrouter"}:
            if not self.valves.OPENAI_BASE_URL:
                raise RuntimeError("OPENAI_BASE_URL не задан для LLM_PROVIDER=openai.")
            if not self.valves.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY не задан для LLM_PROVIDER=openai.")
            base_url = self.valves.OPENAI_BASE_URL.rstrip("/")
            chat_url = (
                base_url
                if base_url.endswith("/chat/completions")
                else f"{base_url}/chat/completions"
            )
            response = requests.post(
                chat_url,
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.valves.LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.1,
                    "stream": False,
                },
                timeout=240,
            )
            response.raise_for_status()
            choices = response.json().get("choices") or []
            if not choices:
                return ""
            return (choices[0].get("message") or {}).get("content", "").strip()

        response = requests.post(
            f"{self.valves.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": self.valves.LLM_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
        return (response.json().get("message") or {}).get("content", "").strip()

    def _stream_llm(self, question: str, context: str):
        messages = self._llm_messages(question, context)
        if self.valves.LLM_PROVIDER.lower() in {"openai", "openai-compatible", "openrouter"}:
            if not self.valves.OPENAI_BASE_URL:
                raise RuntimeError("OPENAI_BASE_URL не задан для LLM_PROVIDER=openai.")
            if not self.valves.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY не задан для LLM_PROVIDER=openai.")
            base_url = self.valves.OPENAI_BASE_URL.rstrip("/")
            chat_url = (
                base_url
                if base_url.endswith("/chat/completions")
                else f"{base_url}/chat/completions"
            )
            with requests.post(
                chat_url,
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.valves.LLM_MODEL,
                    "messages": messages,
                    "temperature": 0.1,
                    "stream": True,
                },
                timeout=240,
                stream=True,
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for choice in payload.get("choices") or []:
                        delta = (choice.get("delta") or {}).get("content")
                        if delta:
                            yield delta
            return

        with requests.post(
            f"{self.valves.OLLAMA_URL.rstrip('/')}/api/chat",
            json={
                "model": self.valves.LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.1},
            },
            timeout=180,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                delta = (payload.get("message") or {}).get("content")
                if delta:
                    yield delta
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
        hits = self._lexically_relevant_hits(question, hits)

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

        context = "\n\n---\n\n".join(context_parts)
        source_lines = [
            f"- {source_file}, {date}, {city}, {event_title}".strip()
            for source_file, date, city, event_title in sources[: self.valves.TOP_K]
        ]
        sources_block = "\n\nИсточники:\n" + "\n".join(source_lines)
        if body.get("stream"):
            def generate():
                for chunk in self._stream_llm(question, context):
                    yield chunk
                yield sources_block
            return generate()
        answer = self._ask_llm(question, context)
        return answer + sources_block


class Pipeline(Pipe):
    pass
