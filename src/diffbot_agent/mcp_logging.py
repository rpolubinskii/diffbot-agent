from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from diffbot_agent.logging_utils import (
    elapsed_ms,
    has_error_marker,
    log_by_verbosity,
    log_event,
    monotonic_ms,
    serialize_for_json,
)


@dataclass
class _PendingToolCall:
    server: str | None
    tool: str
    started_ms: float


class McpRunLogger:
    def __init__(self) -> None:
        self._pending_calls: dict[str, _PendingToolCall] = {}

    def handle_event(self, event: Any) -> None:
        if getattr(event, "type", None) != "run_item_stream_event":
            return

        event_name = getattr(event, "name", None)
        item = getattr(event, "item", None)
        if event_name == "mcp_list_tools":
            self._log_list_tools(item)
        elif event_name == "tool_called":
            self._log_tool_called(item)
        elif event_name == "tool_output":
            self._log_tool_output(item)

    def log_pending_errors(self, error: BaseException) -> None:
        for call_id, pending in list(self._pending_calls.items()):
            log_event(
                "mcp.tool.error",
                {
                    "server": pending.server,
                    "tool": pending.tool,
                    "call_id": call_id,
                    "duration_ms": elapsed_ms(pending.started_ms),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                level=logging.ERROR,
            )
        self._pending_calls.clear()

    def _log_list_tools(self, item: Any) -> None:
        raw_item = getattr(item, "raw_item", None)
        server = _field(raw_item, "server_label")
        error = _field(raw_item, "error")
        tools = _field(raw_item, "tools") or []
        tool_names = [
            name
            for name in (_field(tool, "name") for tool in tools)
            if isinstance(name, str)
        ]
        payload = {
            "server": server,
            "tool_count": len(tool_names),
            "tools": tool_names,
            "error": error,
        }
        if error:
            log_event("mcp.list_tools.error", payload, level=logging.WARNING)
        else:
            log_event("mcp.list_tools.response", payload)

    def _log_tool_called(self, item: Any) -> None:
        origin = getattr(item, "tool_origin", None)
        if not _is_mcp_origin(origin):
            return

        raw_item = getattr(item, "raw_item", None)
        call_id = _call_id(raw_item)
        tool = _field(raw_item, "name") or "<unknown>"
        server = getattr(origin, "mcp_server_name", None)
        arguments = _parse_arguments(_field(raw_item, "arguments"))

        log_by_verbosity(
            debug_event="mcp.tool.request",
            debug_payload={
                "server": server,
                "tool": tool,
                "call_id": call_id,
                "arguments": arguments,
            },
            info_event="mcp.tool.call",
            info_payload={"server": server, "tool": tool},
        )
        if call_id is not None:
            self._pending_calls[call_id] = _PendingToolCall(
                server=server,
                tool=tool,
                started_ms=monotonic_ms(),
            )

    def _log_tool_output(self, item: Any) -> None:
        origin = getattr(item, "tool_origin", None)
        if not _is_mcp_origin(origin):
            return

        raw_item = getattr(item, "raw_item", None)
        call_id = _call_id(raw_item)
        pending = self._pending_calls.pop(call_id, None) if call_id is not None else None
        output = getattr(item, "output", None)
        serialized_output = serialize_for_json(output)
        payload = {
            "server": pending.server if pending is not None else getattr(origin, "mcp_server_name", None),
            "tool": pending.tool if pending is not None else _field(raw_item, "name"),
            "call_id": call_id,
            "duration_ms": elapsed_ms(pending.started_ms) if pending is not None else None,
            "result": serialized_output,
        }
        log_event("mcp.tool.response", payload)
        if has_error_marker(serialized_output):
            log_event("mcp.tool.result_error", payload, level=logging.WARNING)


def _is_mcp_origin(origin: Any) -> bool:
    origin_type = getattr(origin, "type", None)
    value = getattr(origin_type, "value", origin_type)
    return value == "mcp"


def _field(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _call_id(raw_item: Any) -> str | None:
    value = _field(raw_item, "call_id") or _field(raw_item, "id")
    return value if isinstance(value, str) and value else None


def _parse_arguments(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
