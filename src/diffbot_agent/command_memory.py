from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from diffbot_agent.logging_utils import has_error_marker, serialize_for_json


IMAGE_PLACEHOLDER = "[camera image consumed]"
UNKNOWN_OUTPUT_PREVIEW_CHARS = 500
COMPACT_TEXT_PREVIEW_CHARS = 800
STALE_VISUAL_WARNING = (
    "Old visual observations are search hints only and cannot establish current "
    "visibility. Reverify them with a current camera image."
)

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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EventTimestamp:
    elapsed_seconds: float
    observed_at: datetime

    def current_prefix(self) -> str:
        return (
            f"[+{self.elapsed_seconds:.1f}s | "
            f"{self.observed_at.astimezone(timezone.utc):%H:%M:%SZ}]"
        )


@dataclass
class CommandContextState:
    full_tool_rounds: int
    command: str = ""
    robot_status: str = ""
    recent_memories: tuple[CanonicalCommandRecord, ...] = ()
    started_at: str = field(default_factory=utc_now)
    started_monotonic: float = field(default_factory=time.monotonic)
    monotonic_clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    utc_clock: Callable[[], datetime] = field(
        default=lambda: datetime.now(timezone.utc),
        repr=False,
    )
    event_timestamps: dict[str, EventTimestamp] = field(default_factory=dict)
    completed_call_ids: set[str] = field(default_factory=set)
    image_call_ids: set[str] = field(default_factory=set)
    latest_image_call_id: str | None = None
    latest_image_items: tuple[dict[str, Any], dict[str, Any]] | None = None
    latest_image_valid: bool = False

    def __post_init__(self) -> None:
        self._started_datetime = _parse_timestamp(self.started_at)
        self._historical_memory = render_recent_memories(
            list(self.recent_memories),
            now=self._started_datetime,
        )

    def filter_model_input(self, payload: Any) -> Any:
        from agents.run_config import ModelInputData

        current_items = payload.model_data.input
        self._update_visual_state(current_items)
        compacted = compact_current_command(
            current_items,
            self.full_tool_rounds,
        )
        compacted = self._restore_current_image(compacted)
        filtered, visible_image_ids = retain_current_image(
            compacted,
            self.latest_image_call_id if self.latest_image_valid else None,
        )
        self.image_call_ids.update(visible_image_ids)
        filtered = _replace_initial_user_item(
            filtered,
            _render_current_context(
                self.command,
                self.robot_status,
                self._started_datetime,
            ),
        )
        filtered = self._annotate_current_events(filtered)
        filtered.append(_historical_memory_item(self._historical_memory))
        return ModelInputData(
            input=filtered,
            instructions=payload.model_data.instructions,
        )

    def mark_model_request_succeeded(self) -> None:
        pass

    def _update_visual_state(self, items: list[Any]) -> None:
        calls, outputs = _matched_tool_items(items)
        for call, output in calls:
            call_id = _call_id(call)
            if not call_id or call_id in self.completed_call_ids or output is None:
                continue
            self.completed_call_ids.add(call_id)
            tool_name = _normalize_tool_name(
                str(call.get("name") or call.get("type") or "")
            )
            output_value = output.get("output")
            if contains_image(output_value):
                self.latest_image_call_id = call_id
                self.latest_image_items = (copy.deepcopy(call), copy.deepcopy(output))
                self.latest_image_valid = True
            elif _is_environment_changing_tool(tool_name):
                self.latest_image_valid = False

    def _restore_current_image(self, items: list[Any]) -> list[Any]:
        if not self.latest_image_valid or self.latest_image_items is None:
            return items
        if any(
            _item_type(item) in _OUTPUT_TYPES
            and _call_id(item) == self.latest_image_call_id
            for item in items
        ):
            return items
        call, output = self.latest_image_items
        return [*items, copy.deepcopy(call), copy.deepcopy(output)]

    def _annotate_current_events(self, items: list[Any]) -> list[Any]:
        annotated = copy.deepcopy(items)
        initial_end = _initial_input_end(annotated)
        for index, item in enumerate(annotated):
            if index < initial_end or not isinstance(item, dict):
                continue
            item_type = _item_type(item)
            if item_type in _OUTPUT_TYPES:
                call_id = _call_id(item)
                if call_id:
                    prefix = self._event_timestamp(f"tool:{call_id}").current_prefix()
                    item["output"] = _annotate_tool_output(item.get("output"), prefix)
            elif _is_assistant_message(item):
                key = f"message:{_stable_item_key(item)}"
                prefix = self._event_timestamp(key).current_prefix()
                _prefix_message_text(item, prefix)
        return annotated

    def _event_timestamp(self, key: str) -> EventTimestamp:
        existing = self.event_timestamps.get(key)
        if existing is not None:
            return existing
        timestamp = EventTimestamp(
            elapsed_seconds=max(0.0, self.monotonic_clock() - self.started_monotonic),
            observed_at=_as_utc(self.utc_clock()),
        )
        self.event_timestamps[key] = timestamp
        return timestamp


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
    *,
    started_at: str | None = None,
) -> str:
    command_time = _parse_timestamp(started_at or utc_now())
    return (
        _render_current_context(command, robot_status, command_time)
        + "\n"
        + _render_historical_memory_section(
            render_recent_memories(recent_memories, now=command_time)
        )
    )


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


def retain_current_image(
    items: list[Any],
    current_image_call_id: str | None,
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
        if call_id != current_image_call_id:
            item["output"] = sanitize_images(item.get("output"))
    return result, image_call_ids


def replace_consumed_images(
    items: list[Any],
    consumed_call_ids: set[str],
) -> tuple[list[Any], set[str]]:
    """Apply the previous one-shot image policy for compatibility."""
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
    if contains_image(output_value):
        parsed_output = _remove_image_placeholders(parsed_output)
        if parsed_output in (None, "", {}, []):
            return None
    if _is_generic_capture_result(normalized_name, parsed_output):
        return None
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


def _remove_initial_user_item(items: list[Any]) -> list[Any]:
    for index, item in enumerate(items[:_initial_input_end(items)]):
        if isinstance(item, dict) and item.get("role") == "user":
            return [*items[:index], *items[index + 1 :]]
    return items


def _replace_initial_user_item(items: list[Any], text: str) -> list[Any]:
    result = copy.deepcopy(items)
    for index, item in enumerate(result[:_initial_input_end(result)]):
        if isinstance(item, dict) and item.get("role") == "user":
            replacement = copy.deepcopy(item)
            replacement["content"] = text
            result[index] = replacement
            return result
    return [
        {"role": "user", "type": "message", "content": text},
        *result,
    ]


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


def _is_environment_changing_tool(tool_name: str) -> bool:
    return _tool_category(tool_name) in {"navigation", "safety"}


def _is_generic_capture_result(tool_name: str, value: Any) -> bool:
    if not any(marker in tool_name for marker in _VISUAL_MARKERS):
        return False
    serialized = _json_dump(value).lower()
    return any(marker in serialized for marker in _GENERIC_CAPTURE_MARKERS)


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


def _render_current_context(
    command: str,
    robot_status: str,
    started_at: datetime,
) -> str:
    prefix = EventTimestamp(0.0, started_at).current_prefix()
    return (
        f"CURRENT_COMMAND\n{prefix} {command}\n\n"
        f"CURRENT_ROBOT_STATUS\n{prefix} {robot_status}\n\n"
        "CURRENT_COMMAND_TIMELINE"
    )


def _render_historical_memory_section(memory_text: str) -> str:
    return (
        "HISTORICAL_MEMORY\n"
        "Status snapshots and raw images are omitted.\n"
        f"{memory_text}"
    )


def _historical_memory_item(memory_text: str) -> dict[str, Any]:
    return {
        "role": "developer",
        "type": "message",
        "content": _render_historical_memory_section(memory_text),
    }


def _annotate_tool_output(value: Any, prefix: str) -> Any:
    if isinstance(value, str):
        return f"{prefix} {value}"
    if isinstance(value, list):
        return [{"type": "input_text", "text": prefix}, *copy.deepcopy(value)]
    if value is None:
        return prefix
    return f"{prefix} {_json_dump(value)}"


def _prefix_message_text(item: dict[str, Any], prefix: str) -> None:
    content = item.get("content")
    if isinstance(content, str):
        item["content"] = f"{prefix} {content}"
        return
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in {"output_text", "input_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            part["text"] = f"{prefix} {text}"
            return


def _stable_item_key(item: dict[str, Any]) -> str:
    return _json_dump(item)


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


def _remove_image_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(IMAGE_PLACEHOLDER, "").strip()
    if isinstance(value, list):
        cleaned = [_remove_image_placeholders(item) for item in value]
        return [item for item in cleaned if item not in (None, "", {}, [])]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            cleaned_item = _remove_image_placeholders(item)
            if cleaned_item not in (None, "", {}, []):
                cleaned[key] = cleaned_item
        if cleaned.get("type") in {"input_text", "output_text", "text"} and not any(
            key in cleaned for key in ("text", "content")
        ):
            return {}
        return cleaned
    return value


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
