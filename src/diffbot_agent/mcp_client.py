from __future__ import annotations

import logging
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        session = self._require_session()
        started = monotonic_ms()
        log_event("mcp.tool.request", {"tool": name})
        try:
            result = await session.call_tool(name, arguments)
            serialized = serialize_for_json(result)
            if has_error_marker(serialized):
                log_event(
                    "mcp.tool.result_error",
                    {"tool": name, "duration_ms": elapsed_ms(started), "result": serialized},
                    level=logging.WARNING,
                )
            else:
                log_event("mcp.tool.response", {"tool": name, "duration_ms": elapsed_ms(started)})
            return serialized
        except Exception as exc:
            log_event(
                "mcp.tool.error",
                {
                    "tool": name,
                    "duration_ms": elapsed_ms(started),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            raise

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise DiffbotMcpError("MCP client has not been started.")
        return self._session
