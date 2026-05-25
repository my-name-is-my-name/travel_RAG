import os
from pathlib import Path

from dotenv import load_dotenv


CODE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = CODE_ROOT
load_dotenv(CODE_ROOT / ".env")


def resolve_data_root() -> Path:
    raw_value = os.getenv("ROOT_DATA_DIR", "").strip()
    if not raw_value:
        return DEFAULT_DATA_ROOT
    return Path(raw_value).expanduser().resolve()


DATA_ROOT = resolve_data_root()
