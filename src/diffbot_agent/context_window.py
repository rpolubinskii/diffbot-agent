from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from diffbot_agent.episode import (
    _call_id,
    _is_environment_changing_tool,
    _item_type,
    _matched_tool_items,
    _normalize_tool_name,
    _parse_timestamp,
    utc_now,
)
from diffbot_agent.episode import _CALL_TYPES, _OUTPUT_TYPES
from diffbot_agent.sanitize import contains_image, sanitize_images


SERVER_COMPACTION_TYPE = "compaction"


@dataclass
class CommandContextState:
    """Per-turn input shaping. Two robot-specific jobs only: keep the model's
    context to the latest valid camera frame, and bound the length of the retained
    conversation thread on the local (Ollama) path. The thread itself is owned by
    the SDK session; this no longer rebuilds it."""

    max_context_items: int
    compact_locally: bool = True
    completed_call_ids: set[str] = field(default_factory=set)
    image_call_ids: set[str] = field(default_factory=set)
    latest_image_call_id: str | None = None
    latest_image_items: tuple[dict[str, Any], dict[str, Any]] | None = None
    latest_image_valid: bool = False

    def filter_model_input(self, payload: Any) -> Any:
        from agents.run_config import ModelInputData

        current_items = payload.model_data.input
        self._update_visual_state(current_items)
        if self.compact_locally:
            bounded = trim_to_recent(current_items, self.max_context_items)
        else:
            bounded = truncate_at_server_compaction(current_items)
        bounded = self._restore_current_image(bounded)
        filtered, visible_image_ids = retain_current_image(
            bounded,
            self.latest_image_call_id if self.latest_image_valid else None,
        )
        self.image_call_ids.update(visible_image_ids)
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


def compose_command_input(command: str, *, started_at: str | None = None) -> str:
    """The user turn persisted into the session: the command, stamped once at
    arrival so the thread carries its own chronology."""
    stamp = _utc_stamp(_parse_timestamp(started_at or utc_now()))
    return f"{stamp} {command}"


def trim_to_recent(items: list[Any], max_items: int) -> list[Any]:
    """Bound the retained thread on the local path (Ollama has no server-side
    compaction). Keep any leading system/developer preamble plus the most recent
    ``max_items`` body items, then repair the cut so no tool call/output is orphaned.
    Older turns are dropped, not summarized — durable facts live in diffbot-memory."""
    copied = copy.deepcopy(items)
    if max_items <= 0:
        return copied
    head_end = _preamble_end(copied)
    body = copied[head_end:]
    if len(body) <= max_items:
        return copied
    return copied[:head_end] + _repair_tool_pairing(body[-max_items:])


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


def _repair_tool_pairing(items: list[Any]) -> list[Any]:
    """A window cut can orphan tool items: a leading output whose call was dropped,
    or a trailing call whose output was not kept. Both can be rejected by the model
    provider, so trim the window to clean boundaries."""
    call_ids = {_call_id(item) for item in items if _item_type(item) in _CALL_TYPES}
    start = 0
    while start < len(items):
        item = items[start]
        if _item_type(item) in _OUTPUT_TYPES and _call_id(item) not in call_ids:
            start += 1
            continue
        break
    trimmed = items[start:]

    output_ids = {_call_id(item) for item in trimmed if _item_type(item) in _OUTPUT_TYPES}
    end = len(trimmed)
    while end > 0:
        item = trimmed[end - 1]
        if _item_type(item) in _CALL_TYPES and _call_id(item) not in output_ids:
            end -= 1
            continue
        break
    return trimmed[:end]


def _preamble_end(items: list[Any]) -> int:
    """Index past any leading system/developer preamble (kept verbatim by the trim).
    User turns are part of the conversation body and age out with the window."""
    index = 0
    while index < len(items):
        item = items[index]
        if isinstance(item, dict) and item.get("role") in {"system", "developer"}:
            index += 1
            continue
        break
    return index


def _utc_stamp(value: datetime) -> str:
    return f"[{value.astimezone(timezone.utc):%H:%M:%SZ}]"
