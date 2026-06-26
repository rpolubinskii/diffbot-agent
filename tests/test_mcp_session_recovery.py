from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from diffbot_agent.openai_agents_runtime import _mcp_server_class_and_kwargs


@dataclass
class RecordingSession:
    failures: list[BaseException] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any] | None]] = field(default_factory=list)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> str:
        del meta
        self.calls.append((tool_name, arguments))
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def _server():
    from agents.mcp import MCPServerStreamableHttp

    server_class, kwargs = _mcp_server_class_and_kwargs(
        MCPServerStreamableHttp,
        elicitation_callback=None,
    )
    return server_class(
        name="diffbot-mcp",
        params={"url": "http://diffbot-mcp.example/mcp"},
        **kwargs,
    )


def _http_status_error(status_code: int) -> Exception:
    from agents.mcp.server import httpx

    request = httpx.Request("POST", "http://diffbot-mcp.example/mcp")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("server error", request=request, response=response)


async def _install_reconnect(server: Any, replacement_session: RecordingSession) -> list[str]:
    events: list[str] = []

    async def cleanup() -> None:
        events.append("cleanup")
        server.session = None

    async def connect() -> None:
        events.append("connect")
        server.session = replacement_session

    server.cleanup = cleanup
    server.connect = connect
    return events


def test_safe_tool_reconnects_and_retries_once() -> None:
    async def run() -> None:
        server = _server()
        first_session = RecordingSession(failures=[_http_status_error(503)])
        retry_session = RecordingSession()
        server.session = first_session
        events = await _install_reconnect(server, retry_session)

        result = await server.call_tool("robot.get_status", {"include": "all"})

        assert result == "ok"
        assert events == ["cleanup", "connect"]
        assert first_session.calls == [("robot.get_status", {"include": "all"})]
        assert retry_session.calls == [("robot.get_status", {"include": "all"})]

    asyncio.run(run())


def test_unsafe_tool_reconnects_without_retrying() -> None:
    async def run() -> None:
        from agents.exceptions import UserError

        server = _server()
        first_session = RecordingSession(failures=[_http_status_error(503)])
        retry_session = RecordingSession()
        server.session = first_session
        events = await _install_reconnect(server, retry_session)

        with pytest.raises(UserError, match="HTTP error 503"):
            await server.call_tool("nav.move_to", {"x": 1})

        assert events == ["cleanup", "connect"]
        assert first_session.calls == [("nav.move_to", {"x": 1})]
        assert retry_session.calls == []

    asyncio.run(run())


def test_speak_ask_forces_two_minute_timeout() -> None:
    async def run() -> None:
        server = _server()
        session = RecordingSession()
        server.session = session

        result = await server.call_tool(
            "speak.ask",
            {"prompt": "Ready?", "timeoutSeconds": 45.0},
        )

        assert result == "ok"
        assert session.calls == [
            ("speak.ask", {"prompt": "Ready?", "timeoutSeconds": 120.0}),
        ]

    asyncio.run(run())
