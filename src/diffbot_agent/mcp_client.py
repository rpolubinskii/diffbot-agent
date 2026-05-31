from __future__ import annotations

import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


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
        result = await session.read_resource(uri)
        text = _extract_text(result)
        if not text:
            raise DiffbotMcpError(f"MCP resource {uri} returned no text content.")
        return text

    async def command_turn_prompt(self, vocal_command: str, robot_status: str) -> str | None:
        session = self._require_session()
        get_prompt = getattr(session, "get_prompt", None)
        if get_prompt is None:
            return None

        kwargs = {
            "vocal_command": vocal_command,
            "operator_text": "",
            "robot_status": robot_status,
        }

        try:
            result = get_prompt("diffbot.command_turn", arguments=kwargs)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            return None

        return _extract_text(result) or None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise DiffbotMcpError("MCP client has not been started.")
        return self._session


def compose_command_turn(vocal_command: str, robot_status: str) -> str:
    return f"""You are controlling DiffBot for one command turn.

Vocal command:
{vocal_command}

Operator text:

Robot status:
{robot_status}

Rules:
- Prefer high-level diffbot-mcp tools over diagnostics.
- Use diagnostics only when high-level state is unavailable or inconsistent.
- Do not assume images, lidar, raw ROS graph data, or memory search results unless you explicitly call the relevant tool/resource.
- Stop or cancel motion on uncertainty, failed motion, timeout, or interruption.
- Treat backend_unavailable, ros_graph_unavailable, localization_unavailable, navigation_rejected, timeout, and unsafe_request as actionable error classes.
"""


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
