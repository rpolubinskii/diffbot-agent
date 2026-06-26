from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol, runtime_checkable

from diffbot_agent.episode import CanonicalCommandRecord
from diffbot_agent.logging_utils import log_event
from diffbot_agent.mcp_client import DiffbotMcpClient


@runtime_checkable
class MemoryBackend(Protocol):
    """Cross-command memory write seam. Recall is deliberate — the model calls
    diffbot-mcp's ``memory.recall`` tool itself — so backends only persist."""

    async def start(self) -> None: ...

    async def add_episode(self, record: CanonicalCommandRecord) -> None: ...

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


class NullMemoryBackend:
    """No-op backend used when memory is disabled."""

    async def start(self) -> None:
        return None

    async def add_episode(self, record: CanonicalCommandRecord) -> None:
        return None

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        return None


class DiffbotMcpMemoryBackend:
    """Persistent memory via diffbot-mcp's ``memory.remember`` tool (Graphiti behind the
    gateway). Writes are fire-and-forget; recall is deliberate (the model calls
    ``memory.recall`` itself)."""

    def __init__(self, mcp_url: str, client: DiffbotMcpClient | None = None):
        self._client = client if client is not None else DiffbotMcpClient(mcp_url)
        self._connected = False
        self._pending: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        try:
            await self._client.start()
            self._connected = True
        except Exception as exc:
            log_event(
                "memory.connect.error",
                {"error_type": type(exc).__name__, "error": str(exc)},
                level=logging.WARNING,
            )
            self._connected = False

    async def add_episode(self, record: CanonicalCommandRecord) -> None:
        if not self._connected:
            return
        content = json.dumps(record.compact_dict(), ensure_ascii=False)
        task = asyncio.create_task(self._remember(content))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _remember(self, content: str) -> None:
        try:
            await self._client.call_tool("memory.remember", {"content": content})
        except Exception as exc:
            log_event(
                "memory.remember.error",
                {"error_type": type(exc).__name__, "error": str(exc)},
                level=logging.WARNING,
            )

    async def reset(self) -> None:
        # Non-destructive: the persistent graph survives `/reset`. A full wipe is a
        # separate explicit operation.
        return None

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*list(self._pending), return_exceptions=True)
        if self._connected:
            await self._client.stop()
            self._connected = False
