from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from diffbot_agent.logging_utils import (
    elapsed_ms,
    has_error_marker,
    log_event,
    monotonic_ms,
    serialize_for_json,
)


class DiffbotMcpError(RuntimeError):
    pass


@dataclass
class DiffbotMcpClient:
    url: str
    timeout: float = 10.0

    def __post_init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        stack = AsyncExitStack()
        try:
            streams = await stack.enter_async_context(
                streamablehttp_client(self.url, timeout=self.timeout)
            )
            read_stream, write_stream = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self._stack = stack
            self._session = session
        except Exception:
            await stack.aclose()
            raise

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def read_robot_status(self) -> str:
        return await self.read_resource_text("robot://status")

    async def read_resource_text(self, uri: str) -> str:
        session = self._require_session()
        started = monotonic_ms()
        log_event("mcp.resource.request", {"uri": uri})
        try:
            result = await session.read_resource(uri)
            raw_result = serialize_for_json(result)
            if has_error_marker(raw_result):
                log_event(
                    "mcp.resource.result_error",
                    {
                        "uri": uri,
                        "duration_ms": elapsed_ms(started),
                        "raw_result": raw_result,
                    },
                )
            text = _extract_text(result)
            if not text:
                raise DiffbotMcpError(f"MCP resource {uri} returned no text content.")
            log_event(
                "mcp.resource.response",
                {
                    "uri": uri,
                    "duration_ms": elapsed_ms(started),
                    "text": text,
                    "raw_result": raw_result,
                },
            )
            return text
        except Exception as exc:
            log_event(
                "mcp.resource.error",
                {
                    "uri": uri,
                    "duration_ms": elapsed_ms(started),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise DiffbotMcpError("MCP client has not been started.")
        return self._session


def _extract_text(value: Any) -> str:
    parts: list[str] = []
    _collect_text(value, parts)
    return "\n".join(part for part in parts if part).strip()


def _collect_text(value: Any, parts: list[str]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, bytes):
        parts.append(value.decode("utf-8", errors="replace"))
        return
    if isinstance(value, dict):
        for key in ("text", "content", "contents", "messages", "data"):
            if key in value:
                _collect_text(value[key], parts)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_text(item, parts)
        return

    for attr in ("text", "content", "contents", "messages", "data"):
        if hasattr(value, attr):
            _collect_text(getattr(value, attr), parts)
