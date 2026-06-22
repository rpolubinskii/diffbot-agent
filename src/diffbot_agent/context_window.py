from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from diffbot_agent.episode import (
    _as_utc,
    _call_id,
    _is_assistant_message,
    _is_environment_changing_tool,
    _item_type,
    _matched_tool_items,
    _message_text,
    _normalize_tool_name,
    _parse_timestamp,
    compact_tool_event,
    render_historical_memory_section,
    utc_now,
)
from diffbot_agent.episode import _CALL_TYPES, _OUTPUT_TYPES
from diffbot_agent.sanitize import (
    bounded_text as _bounded_text,
    contains_image,
    json_dump as _json_dump,
    sanitize_images,
)


COMPACT_TEXT_PREVIEW_CHARS = 800
SERVER_COMPACTION_TYPE = "compaction"
EMPTY_MEMORY = "(none)"


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
    historical_memory: str = EMPTY_MEMORY
    compact_locally: bool = True
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

    def filter_model_input(self, payload: Any) -> Any:
        from agents.run_config import ModelInputData

        current_items = payload.model_data.input
        self._update_visual_state(current_items)
        if self.compact_locally:
            compacted = compact_current_command(
                current_items,
                self.full_tool_rounds,
            )
        else:
            compacted = truncate_at_server_compaction(current_items)
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
        filtered.append(_historical_memory_item(self.historical_memory))
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


def compose_command_input(
    command: str,
    robot_status: str,
    historical_memory: str,
    *,
    started_at: str | None = None,
) -> str:
    command_time = _parse_timestamp(started_at or utc_now())
    return (
        _render_current_context(command, robot_status, command_time)
        + "\n"
        + render_historical_memory_section(historical_memory)
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


def truncate_at_server_compaction(items: list[Any]) -> list[Any]:
    """Drop items before the latest server ``compaction`` marker (the Responses
    ``context_management`` summary supersedes everything before it)."""
    copied = copy.deepcopy(items)
    last_index: int | None = None
    for index in range(len(copied) - 1, -1, -1):
        if _item_type(copied[index]) == SERVER_COMPACTION_TYPE:
            last_index = index
            break
    if last_index is None:
        return copied
    return copied[last_index:]


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


def _initial_input_end(items: list[Any]) -> int:
    index = 0
    while index < len(items):
        item = items[index]
        if isinstance(item, dict) and item.get("role") in {"user", "system", "developer"}:
            index += 1
            continue
        break
    return index


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


def _historical_memory_item(memory_text: str) -> dict[str, Any]:
    return {
        "role": "developer",
        "type": "message",
        "content": render_historical_memory_section(memory_text),
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
