from __future__ import annotations

import json
import re
from typing import Any

from diffbot_agent.logging_utils import serialize_for_json


IMAGE_PLACEHOLDER = "[camera image consumed]"

_IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
)
_IMAGE_TYPES = {"image", "input_image", "computer_screenshot"}


def contains_image(value: Any) -> bool:
    if isinstance(value, dict):
        if str(value.get("type", "")).lower() in _IMAGE_TYPES:
            return True
        if any(key in value for key in ("image_url", "file_data")):
            return True
        return any(contains_image(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_image(item) for item in value)
    if isinstance(value, str):
        return bool(_IMAGE_DATA_URL_PATTERN.search(value))
    return False


def sanitize_images(value: Any) -> Any:
    if isinstance(value, dict):
        item_type = str(value.get("type", "")).lower()
        if item_type in _IMAGE_TYPES:
            if item_type == "computer_screenshot":
                return {"type": item_type, "image_url": IMAGE_PLACEHOLDER}
            return {"type": "input_text", "text": IMAGE_PLACEHOLDER}

        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"image_url", "file_data"}:
                sanitized[key] = IMAGE_PLACEHOLDER
            else:
                sanitized[key] = sanitize_images(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_images(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_images(item) for item in value]
    if isinstance(value, str):
        return _IMAGE_DATA_URL_PATTERN.sub(IMAGE_PLACEHOLDER, value)
    return value


def sanitize_session_items(items: list[Any]) -> list[Any]:
    import copy

    return [sanitize_images(copy.deepcopy(item)) for item in items]


def remove_image_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(IMAGE_PLACEHOLDER, "").strip()
    if isinstance(value, list):
        cleaned = [remove_image_placeholders(item) for item in value]
        return [item for item in cleaned if item not in (None, "", {}, [])]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = remove_image_placeholders(item)
            if cleaned_item not in (None, "", {}, []):
                cleaned[key] = cleaned_item
        if cleaned.get("type") in {"input_text", "output_text", "text"} and not any(
            key in cleaned for key in ("text", "content")
        ):
            return {}
        return cleaned
    return value


def bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def json_dump(value: Any) -> str:
    return json.dumps(
        sanitize_images(serialize_for_json(value)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
