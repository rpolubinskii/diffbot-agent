from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from diffbot_agent.logging_utils import has_error_marker, serialize_for_json


IMAGE_PLACEHOLDER = "[camera image consumed]"
UNKNOWN_OUTPUT_PREVIEW_CHARS = 500
COMPACT_TEXT_PREVIEW_CHARS = 800

_IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
    re.IGNORECASE,
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
_IMAGE_TYPES = {"image", "input_image", "computer_screenshot"}
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CommandContextState:
    full_tool_rounds: int
    consumed_image_call_ids: set[str] = field(default_factory=set)
    image_call_ids: set[str] = field(default_factory=set)
    pending_image_call_ids: set[str] = field(default_factory=set)

    def filter_model_input(self, payload: Any) -> Any:
        from agents.run_config import ModelInputData

        compacted = compact_current_command(
            payload.model_data.input,
            self.full_tool_rounds,
        )
        filtered, visible_image_ids = replace_consumed_images(
            compacted,
            self.consumed_image_call_ids,
        )
        self.image_call_ids.update(visible_image_ids)
        self.pending_image_call_ids = visible_image_ids - self.consumed_image_call_ids
        return ModelInputData(
            input=filtered,
            instructions=payload.model_data.instructions,
        )

    def mark_model_request_succeeded(self) -> None:
        self.consumed_image_call_ids.update(self.pending_image_call_ids)
        self.pending_image_call_ids.clear()


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
    searchable_text: str
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


class CommandMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            _initialize_command_memory_table(self._connection)

    async def add(self, record: CanonicalCommandRecord) -> None:
        self._add_sync(record)

    def _add_sync(self, record: CanonicalCommandRecord) -> None:
        with self._lock:
            self._require_open()
            self._connection.execute(
                """
                INSERT INTO command_memories (
                    session_id,
                    started_at,
                    completed_at,
                    command,
                    completion_status,
                    final_assistant_text,
                    spoken_text,
                    tool_events,
                    navigation_outcomes,
                    safety_outcomes,
                    error_outcomes,
                    searchable_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.started_at,
                    record.completed_at,
                    record.command,
                    record.completion_status,
                    record.final_assistant_text,
                    _json_dump(list(record.spoken_text)),
                    _json_dump(list(record.tool_events)),
                    _json_dump(list(record.navigation_outcomes)),
                    _json_dump(list(record.safety_outcomes)),
                    _json_dump(list(record.error_outcomes)),
                    record.searchable_text,
                ),
            )
            self._connection.commit()

    async def latest(
        self,
        session_id: str,
        limit: int,
    ) -> list[CanonicalCommandRecord]:
        if limit <= 0:
            return []
        return self._latest_sync(session_id, limit)

    def _latest_sync(
        self,
        session_id: str,
        limit: int,
    ) -> list[CanonicalCommandRecord]:
        with self._lock:
            self._require_open()
            rows = self._connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    started_at,
                    completed_at,
                    command,
                    completion_status,
                    final_assistant_text,
                    spoken_text,
                    tool_events,
                    navigation_outcomes,
                    safety_outcomes,
                    error_outcomes,
                    searchable_text
                FROM command_memories
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [_record_from_row(row) for row in reversed(rows)]

    async def close(self) -> None:
        self._close_sync()

    def _close_sync(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CommandMemoryStore is closed.")


def clear_command_memories(db_path: str | Path, session_id: str) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        _initialize_command_memory_table(connection)
        connection.execute(
            "DELETE FROM command_memories WHERE session_id = ?",
            (session_id,),
        )
        connection.commit()
    finally:
        connection.close()


def compose_command_input(
    command: str,
    robot_status: str,
    recent_memories: list[CanonicalCommandRecord],
) -> str:
    memory_text = render_recent_memories(recent_memories)
    return f"""You are controlling a differential drive robot.

Recent canonical command memory (status snapshots and images are omitted):
{memory_text}

Current operator command:
{command}

Fresh robot://status (authoritative for current pose and state):
{robot_status}

Rules:
- Use the fresh robot://status above instead of remembered pose or telemetry.
- Use speak tool as the main way to communicate with the user.
- Stop or cancel motion on uncertainty, failed motion, timeout, or interruption.
"""


def render_recent_memories(records: list[CanonicalCommandRecord]) -> str:
    if not records:
        return "(none)"

    rendered: list[str] = []
    for record in records:
        item = {
            "completed_at": record.completed_at,
            "command": record.command,
            "completion_status": record.completion_status,
            "final_assistant_text": record.final_assistant_text,
            "spoken_text": list(record.spoken_text),
            "tool_events": list(record.tool_events),
            "navigation_outcomes": list(record.navigation_outcomes),
            "safety_outcomes": list(record.safety_outcomes),
            "error_outcomes": list(record.error_outcomes),
        }
        rendered.append(_json_dump(item))
    return "\n".join(rendered)


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
        events.append(event)
        spoken_text = event.get("spoken_text")
        if isinstance(spoken_text, str) and spoken_text:
            spoken.append(spoken_text)

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
    searchable_payload = {
        "command": command,
        "completion_status": completion_status,
        "final_assistant_text": final_text,
        "spoken_text": spoken,
        "tool_events": events,
        "navigation_outcomes": navigation,
        "safety_outcomes": safety,
        "error_outcomes": errors,
    }
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
        searchable_text=_json_dump(searchable_payload),
    )


def compact_current_command(items: list[Any], full_tool_rounds: int) -> list[Any]:
    copied = copy.deepcopy(items)
    rounds = _completed_tool_rounds(copied)
    # The newest completed round has not been consumed by a model request yet.
    # Even a zero-round policy must pass it through exactly once.
    exact_rounds = max(1, full_tool_rounds) if rounds else 0
    compact_count = max(0, len(rounds) - exact_rounds)
    if compact_count == 0:
        return copied

    compact_end = rounds[compact_count - 1][1]
    preserved_prefix_end = _initial_input_end(copied)
    old_items = copied[preserved_prefix_end:compact_end]
    summary = _compact_items_to_message(old_items)
    result = copied[:preserved_prefix_end]
    if summary is not None:
        result.append(summary)
    result.extend(copied[compact_end:])
    return result


def replace_consumed_images(
    items: list[Any],
    consumed_call_ids: set[str],
) -> tuple[list[Any], set[str]]:
    result = copy.deepcopy(items)
    image_call_ids: set[str] = set()
    for item in result:
        if not isinstance(item, dict) or item.get("type") not in _OUTPUT_TYPES:
            continue
        call_id = _call_id(item)
        if not call_id or not contains_image(item.get("output")):
            continue
        image_call_ids.add(call_id)
        if call_id in consumed_call_ids:
            item["output"] = sanitize_images(item.get("output"))
    return result, image_call_ids


def sanitize_session_items(items: list[Any]) -> list[Any]:
    return [sanitize_images(copy.deepcopy(item)) for item in items]


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


def compact_tool_event(
    call: dict[str, Any],
    output: dict[str, Any] | None,
    *,
    sequence: int,
    fallback_timestamp: str,
) -> dict[str, Any] | None:
    tool_name = str(call.get("name") or call.get("type") or "unknown_tool")
    normalized_name = _normalize_tool_name(tool_name)
    if _is_status_tool(normalized_name):
        return None

    arguments = _parse_json_value(call.get("arguments"))
    output_value = _tool_output_value(output)
    parsed_output = _parse_output(output_value)
    if _is_status_snapshot(parsed_output):
        return None
    status, error_details = _tool_status(parsed_output)
    category = _tool_category(normalized_name)
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


def _completed_tool_rounds(items: list[Any]) -> list[tuple[int, int]]:
    rounds: list[tuple[int, int]] = []
    round_start = _initial_input_end(items)
    index = round_start
    while index < len(items):
        call_indexes: list[int] = []
        call_ids: set[str] = set()
        scan = index
        while scan < len(items):
            item = items[scan]
            item_type = _item_type(item)
            if item_type in _OUTPUT_TYPES and call_ids:
                break
            if item_type in _CALL_TYPES:
                call_id = _call_id(item)
                if call_id:
                    call_ids.add(call_id)
                    call_indexes.append(scan)
            scan += 1

        if not call_ids:
            break

        matched: set[str] = set()
        end = scan
        while end < len(items):
            item = items[end]
            item_type = _item_type(item)
            if item_type in _OUTPUT_TYPES:
                call_id = _call_id(item)
                if call_id in call_ids:
                    matched.add(call_id)
                    if matched == call_ids:
                        rounds.append((round_start, end + 1))
                        round_start = end + 1
                        index = end + 1
                        break
            elif item_type in _CALL_TYPES and matched:
                return rounds
            end += 1
        else:
            break
    return rounds


def _compact_items_to_message(items: list[Any]) -> dict[str, Any] | None:
    calls, outputs = _matched_tool_items(items)
    lines: list[str] = []
    for item in items:
        if _item_type(item) == "reasoning":
            continue
        if _is_assistant_message(item):
            text = _message_text(item)
            if text:
                lines.append(f"Assistant: {_bounded_text(text, COMPACT_TEXT_PREVIEW_CHARS)}")

    for sequence, (call, _output) in enumerate(calls, start=1):
        event = compact_tool_event(
            call,
            outputs.get(_call_id(call)),
            sequence=sequence,
            fallback_timestamp="unknown",
        )
        if event is not None:
            lines.append(f"Tool event: {_json_dump(event)}")

    if not lines:
        return None
    return {
        "role": "assistant",
        "type": "message",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": "Compacted earlier tool rounds:\n" + "\n".join(lines),
            }
        ],
    }


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


def _initial_input_end(items: list[Any]) -> int:
    index = 0
    while index < len(items):
        item = items[index]
        if isinstance(item, dict) and item.get("role") in {"user", "system", "developer"}:
            index += 1
            continue
        break
    return index


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
    if any(marker in tool_name for marker in _SPEECH_MARKERS):
        return "speech"
    if any(marker in tool_name for marker in _SAFETY_MARKERS):
        return "safety"
    if any(marker in tool_name for marker in _NAVIGATION_MARKERS):
        return "navigation"
    return "tool"


def _is_status_tool(tool_name: str) -> bool:
    return any(marker in tool_name for marker in _STATUS_MARKERS)


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


def _bounded_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _json_dump(value: Any) -> str:
    return json.dumps(
        sanitize_images(serialize_for_json(value)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _initialize_command_memory_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS command_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            command TEXT NOT NULL,
            completion_status TEXT NOT NULL,
            final_assistant_text TEXT NOT NULL,
            spoken_text TEXT NOT NULL,
            tool_events TEXT NOT NULL,
            navigation_outcomes TEXT NOT NULL,
            safety_outcomes TEXT NOT NULL,
            error_outcomes TEXT NOT NULL,
            searchable_text TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_command_memories_session_id
        ON command_memories (session_id, id)
        """
    )
    connection.commit()


def _record_from_row(row: tuple[Any, ...]) -> CanonicalCommandRecord:
    return CanonicalCommandRecord(
        record_id=int(row[0]),
        session_id=str(row[1]),
        started_at=str(row[2]),
        completed_at=str(row[3]),
        command=str(row[4]),
        completion_status=str(row[5]),
        final_assistant_text=str(row[6]),
        spoken_text=tuple(json.loads(row[7])),
        tool_events=tuple(json.loads(row[8])),
        navigation_outcomes=tuple(json.loads(row[9])),
        safety_outcomes=tuple(json.loads(row[10])),
        error_outcomes=tuple(json.loads(row[11])),
        searchable_text=str(row[12]),
    )
