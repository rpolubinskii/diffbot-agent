from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
import time
from collections.abc import Mapping
from typing import Any


LOGGER_NAME = "diffbot_agent"

_SECRET_KEY_PATTERN = re.compile(
    r"(^|[_-])("
    r"api[_-]?key|authorization|bearer|password|passwd|secret|access[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|id[_-]?token|auth[_-]?token|session[_-]?token|token"
    r")($|[_-])",
    re.IGNORECASE,
)
_OPENAI_KEY_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+\b", re.IGNORECASE)
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)
_MASK = "[redacted]"
_IMAGE_MASK = "[camera image redacted]"
_IMAGE_TYPES = {"image", "input_image", "computer_screenshot"}


def configure_logging() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(
    event: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: int = logging.INFO,
) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.isEnabledFor(level):
        return

    safe_payload = redact(serialize_for_json(payload or {}))
    logger.log(
        level,
        "%s %s",
        event,
        json.dumps(
            safe_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def monotonic_ms() -> float:
    return time.monotonic() * 1000


def elapsed_ms(start_ms: float) -> int:
    return max(0, round(monotonic_ms() - start_ms))


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        value_type = str(value.get("type", "")).lower()
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY_PATTERN.search(key_text):
                redacted[key_text] = _MASK
            elif key_text in {"image_url", "file_data"}:
                redacted[key_text] = _IMAGE_MASK
            elif value_type in _IMAGE_TYPES and key_text == "data":
                redacted[key_text] = _IMAGE_MASK
            else:
                redacted[key_text] = redact(item)
        return redacted

    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, str):
        redacted = _IMAGE_DATA_URL_PATTERN.sub(_IMAGE_MASK, value)
        return _BEARER_PATTERN.sub("Bearer [redacted]", _OPENAI_KEY_PATTERN.sub(_MASK, redacted))
    return value


def serialize_for_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [serialize_for_json(item) for item in value]
    if dataclasses.is_dataclass(value):
        return serialize_for_json(dataclasses.asdict(value))

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return serialize_for_json(model_dump(mode="json", exclude_unset=True))
        except TypeError:
            return serialize_for_json(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return serialize_for_json(to_dict())

    if hasattr(value, "__dict__"):
        public_attrs = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public_attrs:
            return serialize_for_json(public_attrs)

    return repr(value)


def has_error_marker(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"isError", "is_error"} and bool(item):
                return True
            if key_text == "error" and item:
                return True
            if has_error_marker(item):
                return True
        return False

    if isinstance(value, list):
        return any(has_error_marker(item) for item in value)
    if isinstance(value, tuple):
        return any(has_error_marker(item) for item in value)
    return False
