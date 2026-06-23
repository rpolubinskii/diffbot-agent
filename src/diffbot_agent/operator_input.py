from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Final


class OperatorInputRoute(Enum):
    COMMAND = "command"
    ELICITATION = "elicitation"
    CLOSED = "closed"


_CLOSED: Final = object()


class OperatorInputCoordinator:
    def __init__(self) -> None:
        self._commands: asyncio.Queue[str | object] = asyncio.Queue()
        self._pending_answer: asyncio.Future[str | None] | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def submit(self, text: str) -> OperatorInputRoute:
        route = await self.submit_answer(text)
        if route is OperatorInputRoute.COMMAND:
            return await self.enqueue_command(text)
        return route

    async def submit_answer(self, text: str) -> OperatorInputRoute:
        value = text.strip()
        if not value:
            return OperatorInputRoute.CLOSED if self._closed else OperatorInputRoute.COMMAND

        async with self._lock:
            if self._closed:
                return OperatorInputRoute.CLOSED
            if self._pending_answer is not None and not self._pending_answer.done():
                self._pending_answer.set_result(value)
                return OperatorInputRoute.ELICITATION

        return OperatorInputRoute.COMMAND

    async def enqueue_command(self, text: str) -> OperatorInputRoute:
        value = text.strip()
        if not value:
            return OperatorInputRoute.CLOSED if self._closed else OperatorInputRoute.COMMAND
        async with self._lock:
            if self._closed:
                return OperatorInputRoute.CLOSED
        await self._commands.put(value)
        return OperatorInputRoute.COMMAND

    async def request_answer(self, timeout_seconds: float) -> str | None:
        async with self._lock:
            if self._closed:
                return None
            if self._pending_answer is not None and not self._pending_answer.done():
                raise RuntimeError("Another elicitation request is already pending.")
            pending = asyncio.get_running_loop().create_future()
            self._pending_answer = pending

        try:
            return await asyncio.wait_for(pending, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            if not pending.done():
                pending.cancel()
            return None
        except asyncio.CancelledError:
            if not pending.done():
                pending.cancel()
            raise
        finally:
            async with self._lock:
                if self._pending_answer is pending:
                    self._pending_answer = None

    async def next_command(self) -> str | None:
        item = await self._commands.get()
        if item is _CLOSED:
            return None
        return str(item)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._pending_answer is not None and not self._pending_answer.done():
                self._pending_answer.set_result(None)
        await self._commands.put(_CLOSED)


ElicitationAnswerProvider = Callable[[float], Awaitable[str | None]]
