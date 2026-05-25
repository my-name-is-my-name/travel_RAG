import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import frontmatter
import requests
import yaml
from dotenv import load_dotenv

from ingest import ingest_file
from path_config import CODE_ROOT, DATA_ROOT


RAW_DIR = DATA_ROOT / "raw_notes"
STRUCTURED_DIR = DATA_ROOT / "structured_notes"
ERRORS_DIR = CODE_ROOT / "errors"
STATE_DIR = CODE_ROOT / ".state"
HASHES_PATH = STATE_DIR / "hashes.json"
PROMPT_PATH = CODE_ROOT / "formatter_prompt.md"
DEFAULTS = {
    "QDRANT_URL": "http://localhost:6333",
    "OLLAMA_URL": "http://localhost:11434",
    "EMBEDDING_MODEL": "bge-m3",
    "COLLECTION": "travel_notes",
    "FORMATTER_PROVIDER": "ollama",
    "FORMATTER_MODEL": "gemma3:4b",
    "OPENAI_BASE_URL": "",
    "OPENAI_API_KEY": "",
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


def ensure_dirs():
    RAW_DIR.mkdir(exist_ok=True)
    STRUCTURED_DIR.mkdir(exist_ok=True)
    ERRORS_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)


def load_hashes():
    if not HASHES_PATH.exists():
        return {}
    with HASHES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_hashes(hashes):
    with HASHES_PATH.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2, sort_keys=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def call_formatter(raw_text, source_file, config):
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    provider = config["FORMATTER_PROVIDER"].strip().lower()
    model = config["FORMATTER_MODEL"]
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                f"source_file: {source_file}\n\n"
                "Исходная заметка:\n\n"
                f"{raw_text}"
            ),
        },
    ]
    if provider == "openai":
        base_url = config["OPENAI_BASE_URL"].rstrip("/")
        api_key = config["OPENAI_API_KEY"].strip()
        if not base_url or not api_key:
            raise RuntimeError(
                "Formatter provider=openai, но не заданы OPENAI_BASE_URL или OPENAI_API_KEY."
            )
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=600,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Formatter model недоступна: {exc}. Проверьте {base_url} и модель {model}."
            ) from exc
        data = response.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "").strip()
    else:
        base_url = config["OLLAMA_URL"].rstrip("/")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            response = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Formatter model недоступна: {exc}. Проверьте {base_url} и модель {model}."
            ) from exc
        data = response.json()
        content = (data.get("message") or {}).get("content", "").strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("markdown"):
            content = content[len("markdown") :].strip()
    return content


def looks_like_markdown(text):
    if not text or len(text.strip()) < 20:
        return False
    return text.lstrip().startswith("---") and "#" in text


def set_source_file(markdown_text, source_file):
    post = frontmatter.loads(markdown_text)
    post.metadata["source_file"] = source_file
    return frontmatter.dumps(post).strip() + "\n"


def write_error(raw_path, message, raw_response=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    error_path = ERRORS_DIR / f"{raw_path.stem}.{timestamp}.error.md"
    body = [
        f"# Ошибка обработки {raw_path.name}",
        "",
        message,
        "",
    ]
    if raw_response:
        body.extend(["## Ответ formatter", "", "```markdown", raw_response, "```", ""])
    error_path.write_text("\n".join(body), encoding="utf-8")
    return error_path


def resolve_raw_file(file_arg):
    path = Path(file_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == RAW_DIR.name:
        return DATA_ROOT / path
    return RAW_DIR / path


def resolve_structured_file(file_arg):
    path = Path(file_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == STRUCTURED_DIR.name:
        return DATA_ROOT / path
    if path.parts and path.parts[0] == RAW_DIR.name:
        return STRUCTURED_DIR / path.name
    return STRUCTURED_DIR / path


def iter_raw_files(single_file=None):
    if single_file:
        return [resolve_raw_file(single_file)]
    return sorted(RAW_DIR.glob("*.md"))


def iter_structured_files(single_file=None):
    if single_file:
        return [resolve_structured_file(single_file)]
    return sorted(STRUCTURED_DIR.glob("*.md"))


def process_structured_file(structured_path, config):
    if not structured_path.exists():
        raise FileNotFoundError(f"Structured-файл не найден: {structured_path}")
    chunks = ingest_file(structured_path, config, force=True)
    print(f"Indexed {chunks} chunks: {structured_path.name}")
    return True


def process_file(raw_path, hashes, config, force=False, skip_format=False):
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw-файл не найден: {raw_path}")

    source_file = raw_path.name
    structured_path = STRUCTURED_DIR / source_file
    current_hash = sha256_file(raw_path)
    previous_hash = hashes.get(source_file)

    if not force and previous_hash == current_hash and not skip_format:
        print(f"Skipped unchanged: {source_file}")
        return False

    if skip_format:
        if not structured_path.exists():
            raise FileNotFoundError(
                f"--skip-format: нет structured-файла для {source_file}: {structured_path}"
            )
    else:
        raw_text = raw_path.read_text(encoding="utf-8")
        formatted = call_formatter(raw_text, source_file, config)
        if not looks_like_markdown(formatted):
            error_path = write_error(raw_path, "Formatter вернул пустой ответ или не Markdown.", formatted)
            print(f"Formatter error saved: {error_path}")
            return False
        try:
            formatted = set_source_file(formatted, source_file)
        except Exception as exc:
            error_path = write_error(raw_path, f"Не удалось прочитать YAML frontmatter: {exc}", formatted)
            print(f"Formatter error saved: {error_path}")
            return False
        structured_path.write_text(formatted, encoding="utf-8")
        print(f"Formatted: {source_file} -> {structured_path}")

    chunks = ingest_file(structured_path, config, force=True)
    hashes[source_file] = current_hash
    print(f"Indexed {chunks} chunks: {source_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync raw travel notes to structured notes and Qdrant.")
    parser.add_argument("--file", help="Path to one raw markdown file, usually raw_notes/name.md.")
    parser.add_argument("--force", action="store_true", help="Regenerate and reindex all selected notes.")
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Do not call formatter; index existing structured_notes files.",
    )
    args = parser.parse_args()

    ensure_dirs()
    config = load_config()
    hashes = load_hashes()

    processed = 0
    if args.skip_format:
        for structured_path in iter_structured_files(args.file):
            if process_structured_file(structured_path, config):
                processed += 1
    else:
        for raw_path in iter_raw_files(args.file):
            if process_file(raw_path, hashes, config, force=args.force, skip_format=False):
                processed += 1

    save_hashes(hashes)
    print(f"Done. Processed files: {processed}")


if __name__ == "__main__":
    main()
