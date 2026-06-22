from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from diffbot_agent.logging_utils import has_error_marker, serialize_for_json
from diffbot_agent.sanitize import (
    bounded_text as _bounded_text,
    contains_image,
    json_dump as _json_dump,
    remove_image_placeholders as _remove_image_placeholders,
    sanitize_images,
)


UNKNOWN_OUTPUT_PREVIEW_CHARS = 500
STALE_VISUAL_WARNING = (
    "Old visual observations are search hints only and cannot establish current "
    "visibility. Reverify them with a current camera image."
)

_CALL_TYPES = {
    "function_call",
    "computer_call",
    "custom_tool_call",
    "local_shell_call",
    "shell_call",
}
_OUTPUT_TYPES = {
    "function_call_output",
    "computer_call_output",
    "custom_tool_call_output",
    "local_shell_call_output",
    "shell_call_output",
}
_SPEECH_MARKERS = ("speak", "say", "speech")
_NAVIGATION_MARKERS = (
    "nav_",
    "navigate",
    "drive",
    "move",
    "spin",
    "turn",
    "follow",
    "dock",
)
_SAFETY_MARKERS = ("stop", "cancel", "safety", "emergency", "estop")
_STATUS_MARKERS = ("status", "robot_state", "get_pose", "telemetry")
_VISUAL_MARKERS = ("camera", "image", "photo", "snapshot", "vision")
_GENERIC_CAPTURE_MARKERS = (
    "camera image captured",
    "image captured",
    "photo captured",
    "snapshot captured",
    "capture successful",
)
_IDENTIFIER_KEYS = {
    "id",
    "call_id",
    "goal_id",
    "ros_goal_uuid",
    "response_id",
    "request_id",
    "action",
    "topic",
}

TOOL_CATEGORIES = {"speech", "navigation", "safety", "status", "vision", "tool"}

# Categories come from diffbot-mcp tool _meta (set_mcp_tool_categories); the
# substring heuristic in _tool_category is only a fallback for tools without it.
_MCP_TOOL_CATEGORIES: dict[str, str] = {}
_TOOL_CATEGORY_OVERRIDES: dict[str, str] = {}


def set_mcp_tool_categories(mapping: dict[str, str]) -> None:
    """Install categories advertised by diffbot-mcp tool ``_meta`` (the source of truth)."""
    _MCP_TOOL_CATEGORIES.clear()
    _MCP_TOOL_CATEGORIES.update(
        {_normalize_tool_name(name): category for name, category in mapping.items()}
    )


def set_tool_category_overrides(mapping: dict[str, str]) -> None:
    """Install operator-supplied tool->category overrides (from ``[tool_categories]``)."""
    _TOOL_CATEGORY_OVERRIDES.clear()
    _TOOL_CATEGORY_OVERRIDES.update(
        {_normalize_tool_name(name): category for name, category in mapping.items()}
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CanonicalCommandRecord:
    session_id: str
    started_at: str
    completed_at: str
    command: str
    completion_status: str
    final_assistant_text: str
    spoken_text: tuple[str, ...]
    tool_events: tuple[dict[str, Any], ...]
    navigation_outcomes: tuple[dict[str, Any], ...]
    safety_outcomes: tuple[dict[str, Any], ...]
    error_outcomes: tuple[dict[str, Any], ...]
    record_id: int | None = None

    def compact_dict(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "command": self.command,
            "completion_status": self.completion_status,
            "final_assistant_text": self.final_assistant_text,
            "spoken_text": list(self.spoken_text),
            "tool_events": list(self.tool_events),
            "navigation_outcomes": list(self.navigation_outcomes),
            "safety_outcomes": list(self.safety_outcomes),
            "error_outcomes": list(self.error_outcomes),
        }


def render_recent_memories(
    records: list[CanonicalCommandRecord],
    *,
    now: datetime | None = None,
) -> str:
    if not records:
        return "(none)"

    rendered_at = _as_utc(now or datetime.now(timezone.utc))
    rendered: list[str] = []
    stale_hints: list[str] = []
    seen_text: set[str] = set()
    for record in records:
        observed_at = _parse_timestamp(record.completed_at)
        prefix = _historical_prefix(observed_at, rendered_at)
        current_outcomes: list[str] = []
        for text in _canonical_record_texts(record):
            normalized = _normalize_text(text)
            if not normalized or normalized in seen_text:
                continue
            seen_text.add(normalized)
            if _is_scene_dependent_conclusion(text):
                stale_hints.append(f"{prefix} Observation: {text}")
            else:
                current_outcomes.append(text)

        item = {
            "command": record.command,
            "completion_status": record.completion_status,
            "outcomes": current_outcomes,
            "tool_events": _canonical_events(record),
        }
        rendered.append(f"{prefix} {_json_dump(item)}")

    if stale_hints:
        rendered.extend(
            [
                "",
                "STALE_HINTS_REQUIRING_REVERIFICATION",
                STALE_VISUAL_WARNING,
                *stale_hints,
            ]
        )
    return "\n".join(rendered)


def render_historical_memory_section(memory_text: str) -> str:
    return (
        "HISTORICAL_MEMORY\n"
        "Status snapshots and raw images are omitted.\n"
        f"{memory_text}"
    )


def build_canonical_record(
    *,
    session_id: str,
    started_at: str,
    completed_at: str,
    command: str,
    completion_status: str,
    items: list[Any],
    final_output: Any = None,
    error: BaseException | None = None,
) -> CanonicalCommandRecord:
    serialized_items = [serialize_for_json(item) for item in items]
    calls, outputs = _matched_tool_items(serialized_items)
    events: list[dict[str, Any]] = []
    spoken: list[str] = []

    for sequence, (call, output) in enumerate(calls, start=1):
        event = compact_tool_event(
            call,
            outputs.get(_call_id(call)),
            sequence=sequence,
            fallback_timestamp=completed_at,
        )
        if event is None:
            continue
        spoken_text = event.get("spoken_text")
        if isinstance(spoken_text, str) and spoken_text:
            spoken.append(spoken_text)
        if event.get("category") != "speech":
            events.append(event)

    navigation = tuple(
        event for event in events if event.get("category") == "navigation"
    )
    safety = tuple(event for event in events if event.get("category") == "safety")
    errors = [event for event in events if event.get("status") == "error"]
    if error is not None:
        errors.append(
            {
                "category": "runtime",
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "timestamp": completed_at,
            }
        )

    final_text = _final_assistant_text(serialized_items, final_output)
    spoken = _deduplicate_texts(spoken, excluded=(final_text,))
    return CanonicalCommandRecord(
        session_id=session_id,
        started_at=started_at,
        completed_at=completed_at,
        command=command,
        completion_status=completion_status,
        final_assistant_text=final_text,
        spoken_text=tuple(spoken),
        tool_events=tuple(events),
        navigation_outcomes=navigation,
        safety_outcomes=safety,
        error_outcomes=tuple(errors),
    )


def compact_tool_event(
    call: dict[str, Any],
    output: dict[str, Any] | None,
    *,
    sequence: int,
    fallback_timestamp: str,
) -> dict[str, Any] | None:
    tool_name = str(call.get("name") or call.get("type") or "unknown_tool")
    normalized_name = _normalize_tool_name(tool_name)
    category = _tool_category(normalized_name)
    if category == "status":
        return None

    arguments = _parse_json_value(call.get("arguments"))
    output_value = _tool_output_value(output)
    parsed_output = _parse_output(output_value)
    if contains_image(output_value):
        parsed_output = _remove_image_placeholders(parsed_output)
        if parsed_output in (None, "", {}, []):
            return None
    if _is_generic_capture_result(category, parsed_output):
        return None
    if _is_status_snapshot(parsed_output):
        return None
    status, error_details = _tool_status(parsed_output)
    timestamp = _find_first_key(parsed_output, "timestamp") or fallback_timestamp
    identifiers = _collect_identifiers(parsed_output)

    event: dict[str, Any] = {
        "sequence": sequence,
        "tool": tool_name,
        "operation": normalized_name,
        "category": category,
        "status": status,
        "timestamp": timestamp,
    }
    call_id = _call_id(call)
    if call_id:
        event["call_id"] = call_id

    if category == "speech":
        spoken_text = _find_spoken_text(arguments)
        if spoken_text:
            event["spoken_text"] = spoken_text
        if error_details:
            event["error"] = error_details
        return event

    if arguments not in (None, {}, []):
        event["arguments"] = sanitize_images(arguments)
    if identifiers:
        event["identifiers"] = identifiers
    if category in {"navigation", "safety"}:
        outcome = _compact_outcome(parsed_output)
        if outcome:
            event["outcome"] = outcome
    elif parsed_output not in (None, "", {}, []):
        event["output_preview"] = _bounded_preview(parsed_output)
    if error_details:
        event["error"] = error_details
    return event


def matched_tool_items(
    items: list[Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], dict[str, dict[str, Any]]]:
    return _matched_tool_items(items)


def _matched_tool_items(
    items: list[Any],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any] | None]], dict[str, dict[str, Any]]]:
    outputs: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and _item_type(item) in _OUTPUT_TYPES:
            call_id = _call_id(item)
            if call_id:
                outputs[call_id] = item

    calls: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    for item in items:
        if not isinstance(item, dict) or _item_type(item) not in _CALL_TYPES:
            continue
        call_id = _call_id(item)
        if call_id and call_id in outputs:
            calls.append((item, outputs[call_id]))
    return calls, outputs


def _final_assistant_text(items: list[Any], final_output: Any) -> str:
    if isinstance(final_output, str) and final_output.strip():
        return final_output.strip()

    for item in reversed(items):
        if _is_assistant_message(item):
            text = _message_text(item)
            if text:
                return text
    return ""


def _message_text(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    content = item.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") in {
            "output_text",
            "input_text",
            "text",
        }:
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part).strip()


def _is_assistant_message(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("role") == "assistant"
        and item.get("type", "message") == "message"
    )


def _item_type(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("type", ""))
    return ""


def _call_id(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    value = item.get("call_id") or item.get("id")
    return value if isinstance(value, str) else ""


def _normalize_tool_name(tool_name: str) -> str:
    normalized = tool_name.lower()
    if "__" in normalized:
        normalized = normalized.rsplit("__", 1)[-1]
    return normalized.replace(".", "_").replace("-", "_")


def _tool_category(tool_name: str) -> str:
    override = _TOOL_CATEGORY_OVERRIDES.get(tool_name)
    if override:
        return override
    provided = _MCP_TOOL_CATEGORIES.get(tool_name)
    if provided:
        return provided
    # Last-resort heuristic for tools that ship without category meta.
    if any(marker in tool_name for marker in _SPEECH_MARKERS):
        return "speech"
    if any(marker in tool_name for marker in _SAFETY_MARKERS):
        return "safety"
    if any(marker in tool_name for marker in _NAVIGATION_MARKERS):
        return "navigation"
    if any(marker in tool_name for marker in _STATUS_MARKERS):
        return "status"
    if any(marker in tool_name for marker in _VISUAL_MARKERS):
        return "vision"
    return "tool"


def _is_environment_changing_tool(tool_name: str) -> bool:
    return _tool_category(tool_name) in {"navigation", "safety"}


def _is_generic_capture_result(category: str, value: Any) -> bool:
    if category != "vision":
        return False
    serialized = _json_dump(value).lower()
    return any(marker in serialized for marker in _GENERIC_CAPTURE_MARKERS)


def _is_status_snapshot(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    snapshot_keys = {
        "pose",
        "current_velocity",
        "imu",
        "telemetry",
        "battery",
        "motor_safety",
    }
    return len(snapshot_keys.intersection(value)) >= 2 or "telemetry" in value


def _parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return serialize_for_json(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _tool_output_value(output: dict[str, Any] | None) -> Any:
    if not output:
        return None
    return output.get("output")


def _parse_output(value: Any) -> Any:
    value = sanitize_images(serialize_for_json(value))
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, list):
        text_parts = [
            item.get("text")
            for item in value
            if isinstance(item, dict)
            and item.get("type") in {"input_text", "output_text", "text"}
            and isinstance(item.get("text"), str)
        ]
        if len(text_parts) == 1:
            return _parse_output(text_parts[0])
    return value


def _tool_status(output: Any) -> tuple[str, str]:
    if output is None:
        return "unknown", ""
    if isinstance(output, str) and any(
        marker in output.lower()
        for marker in ("error", "failed", "failure", "exception", "aborted")
    ):
        return "error", _bounded_text(output, UNKNOWN_OUTPUT_PREVIEW_CHARS)
    if has_error_marker(output):
        return "error", _extract_error(output)
    if isinstance(output, dict):
        if output.get("ok") is False or output.get("success") is False:
            return "error", _extract_error(output)
        ros = output.get("ros")
        if isinstance(ros, dict) and ros.get("success") is False:
            return "error", _extract_error(ros)
        state = str(output.get("state") or "").lower()
        if state in {"failed", "aborted", "canceled", "cancelled", "rejected"}:
            return "error", _extract_error(output)
        nested_state = _find_first_key(output, "state").lower()
        if nested_state in {"failed", "aborted", "canceled", "cancelled", "rejected"}:
            return "error", _extract_error(output)
    return "completed", ""


def _extract_error(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("error", "message", "status_text", "detail", "reason"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return _bounded_text(item, UNKNOWN_OUTPUT_PREVIEW_CHARS)
        for item in value.values():
            nested = _extract_error(item)
            if nested:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _extract_error(item)
            if nested:
                return nested
    return "Tool reported an error."


def _find_spoken_text(arguments: Any) -> str:
    if isinstance(arguments, dict):
        for key in ("text", "message", "speech"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(arguments, str):
        return arguments.strip()
    return ""


def _find_first_key(value: Any, key: str) -> str:
    if isinstance(value, dict):
        item = value.get(key)
        if isinstance(item, (str, int, float)):
            return str(item)
        for nested in value.values():
            found = _find_first_key(nested, key)
            if found:
                return found
    if isinstance(value, list):
        for nested in value:
            found = _find_first_key(nested, key)
            if found:
                return found
    return ""


def _collect_identifiers(value: Any) -> dict[str, Any]:
    identifiers: dict[str, Any] = {}

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if (
                    key in _IDENTIFIER_KEYS or key.endswith("_id")
                ) and isinstance(nested, (str, int)):
                    identifiers.setdefault(key, nested)
                else:
                    collect(nested)
        elif isinstance(item, list):
            for nested in item:
                collect(nested)

    collect(value)
    return identifiers


def _compact_outcome(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    outcome: dict[str, Any] = {}
    for key in ("ok", "success", "state", "status", "status_text", "action"):
        if key in value and isinstance(value[key], (str, int, float, bool)):
            outcome[key] = value[key]
    ros = value.get("ros")
    if isinstance(ros, dict):
        for key in ("success", "state", "status", "status_text", "action"):
            if key in ros and isinstance(ros[key], (str, int, float, bool)):
                outcome[f"ros_{key}"] = ros[key]
    return outcome


def _bounded_preview(value: Any) -> str:
    return _bounded_text(_json_dump(sanitize_images(value)), UNKNOWN_OUTPUT_PREVIEW_CHARS)


def _canonical_record_texts(record: CanonicalCommandRecord) -> list[str]:
    texts = [record.final_assistant_text, *record.spoken_text]
    for event in record.tool_events:
        for key in ("spoken_text", "output_preview"):
            value = event.get(key)
            if isinstance(value, str):
                texts.append(value)
    return _deduplicate_texts(texts)


def _canonical_events(record: CanonicalCommandRecord) -> list[dict[str, Any]]:
    events = [
        *record.tool_events,
        *record.navigation_outcomes,
        *record.safety_outcomes,
        *record.error_outcomes,
    ]
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        if event.get("category") == "speech":
            continue
        cleaned = _remove_image_placeholders(event)
        if not isinstance(cleaned, dict):
            continue
        cleaned.pop("spoken_text", None)
        cleaned.pop("output_preview", None)
        if not cleaned:
            continue
        key = _json_dump(cleaned)
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _deduplicate_texts(
    texts: list[str],
    *,
    excluded: tuple[str, ...] = (),
) -> list[str]:
    seen = {_normalize_text(text) for text in excluded if text}
    result: list[str] = []
    for text in texts:
        stripped = text.strip()
        normalized = _normalize_text(stripped)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(stripped)
    return result


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _is_scene_dependent_conclusion(text: str) -> bool:
    normalized = f" {_normalize_text(text)} "
    visual_terms = (
        " i see ",
        " visible ",
        " appears ",
        " located ",
        " location ",
        " to the left ",
        " to the right ",
        " in front ",
        " behind ",
        " next to ",
        " near ",
        " beside ",
        " by the ",
        " under ",
        " above ",
        " inside ",
        " outside ",
        " on the ",
        " is in ",
        " is at ",
        " is on ",
        " was in ",
        " was at ",
        " was on ",
    )
    return any(term in normalized for term in visual_terms)


def _historical_prefix(observed_at: datetime, now: datetime) -> str:
    age_seconds = max(0, int((now - observed_at).total_seconds()))
    return f"[{observed_at:%H:%M:%SZ} | age {_compact_age(age_seconds)}]"


def _compact_age(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
