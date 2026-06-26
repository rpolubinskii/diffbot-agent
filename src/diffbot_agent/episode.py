from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from diffbot_agent.logging_utils import serialize_for_json
from diffbot_agent.sanitize import (
    bounded_text as _bounded_text,
    json_dump as _json_dump,
    remove_image_placeholders as _remove_image_placeholders,
    sanitize_images,
)


UNKNOWN_OUTPUT_PREVIEW_CHARS = 500

# Conversation-item types for tool turns in the OpenAI Agents SDK transcript. This is
# coupling to the model/SDK item format (not to diffbot-mcp): the agent must tell a
# tool call from its output to shape its own context. MCP tools surface as function calls.
_CALL_TYPES = {"function_call"}
_OUTPUT_TYPES = {"function_call_output"}

# Tool categories are owned by diffbot-mcp (advertised in each tool's _meta) and
# loaded once at startup. The agent never classifies tools itself.
_MCP_TOOL_CATEGORIES: dict[str, str] = {}


def set_mcp_tool_categories(mapping: dict[str, str]) -> None:
    """Install categories advertised by diffbot-mcp tool ``_meta`` (the single source of truth)."""
    _MCP_TOOL_CATEGORIES.clear()
    _MCP_TOOL_CATEGORIES.update(
        {_normalize_tool_name(name): category for name, category in mapping.items()}
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CanonicalCommandRecord:
    """A distilled, mcp-agnostic episode written to long-term memory after each command.
    Tool outputs are kept opaque — the memory service's LLM extracts meaning from them."""

    session_id: str
    started_at: str
    completed_at: str
    command: str
    completion_status: str
    final_assistant_text: str
    tool_events: tuple[dict[str, Any], ...]
    error: str = ""

    def compact_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "command": self.command,
            "completion_status": self.completion_status,
            "final_assistant_text": self.final_assistant_text,
            "tool_events": list(self.tool_events),
        }
        if self.error:
            data["error"] = self.error
        return data


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
    for sequence, (call, _output) in enumerate(calls, start=1):
        event = compact_tool_event(call, outputs.get(_call_id(call)), sequence=sequence)
        if event is not None:
            events.append(event)
    return CanonicalCommandRecord(
        session_id=session_id,
        started_at=started_at,
        completed_at=completed_at,
        command=command,
        completion_status=completion_status,
        final_assistant_text=_final_assistant_text(serialized_items, final_output),
        tool_events=tuple(events),
        error=str(error) if error is not None else "",
    )


def compact_tool_event(
    call: dict[str, Any],
    output: dict[str, Any] | None,
    *,
    sequence: int,
) -> dict[str, Any] | None:
    """One opaque tool record: name, the mcp-provided category, arguments, and a bounded
    text preview of the output. The output is not parsed or interpreted."""
    tool_name = str(call.get("name") or call.get("type") or "unknown_tool")
    category = _tool_category(_normalize_tool_name(tool_name))
    if category == "status":
        return None  # status reads are noise in long-term memory

    event: dict[str, Any] = {"sequence": sequence, "tool": tool_name, "category": category}
    arguments = _loads(call.get("arguments"))
    if arguments not in (None, "", {}, []):
        event["arguments"] = sanitize_images(arguments)
    preview = _output_preview(output)
    if preview:
        event["output"] = preview
    return event


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
    # diffbot-mcp owns categories; a tool that advertises none falls back to the generic bucket.
    return _MCP_TOOL_CATEGORIES.get(tool_name, "tool")


def _is_environment_changing_tool(tool_name: str) -> bool:
    return _tool_category(tool_name) in {"navigation", "safety"}


def _loads(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return serialize_for_json(value)


def _output_preview(output: dict[str, Any] | None) -> str:
    if not output:
        return ""
    value = _remove_image_placeholders(sanitize_images(serialize_for_json(output.get("output"))))
    if value in (None, "", {}, []):
        return ""
    text = value if isinstance(value, str) else _json_dump(value)
    return _bounded_text(text, UNKNOWN_OUTPUT_PREVIEW_CHARS)


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
