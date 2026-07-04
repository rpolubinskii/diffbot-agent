from __future__ import annotations

import asyncio

from dataclasses import dataclass

from diffbot_agent.agent_runtime import AgentRuntime, TurnResult
from diffbot_agent.episode import utc_now


@dataclass
class RuntimeController:
    runtime: AgentRuntime

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.runtime.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        try:
            await self.runtime.stop()
        finally:
            self._started = False

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float = 600.0,
    ) -> TurnResult:
        normalized = command.strip()
        if not normalized:
            return TurnResult(
                status="failed",
                started_at=utc_now(),
                completed_at=utc_now(),
                error="command must not be empty",
            )

        if self._lock.locked():
            return TurnResult(
                status="busy",
                started_at=utc_now(),
                completed_at=utc_now(),
                error="diffbot-agent is already running a command",
            )

        async with self._lock:
            started_at = utc_now()
            try:
                return await asyncio.wait_for(
                    self.runtime.run_turn(normalized),
                    timeout=max(0.1, float(timeout_seconds)),
                )
            except TimeoutError:
                completed_at = utc_now()
                return TurnResult(
                    status="timeout",
                    started_at=started_at,
                    completed_at=completed_at,
                    error=f"command timed out after {timeout_seconds:g} seconds",
                )
            except Exception as exc:
                completed_at = utc_now()
                return TurnResult(
                    status="failed",
                    started_at=started_at,
                    completed_at=completed_at,
                    error=f"{type(exc).__name__}: {exc}",
                )

    async def reset(self) -> TurnResult:
        if self._lock.locked():
            return TurnResult(
                status="busy",
                started_at=utc_now(),
                completed_at=utc_now(),
                error="diffbot-agent is already running a command",
            )

        async with self._lock:
            started_at = utc_now()
            try:
                await self.runtime.reset()
            except Exception as exc:
                return TurnResult(
                    status="failed",
                    started_at=started_at,
                    completed_at=utc_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            return TurnResult(
                status="completed",
                final_text="session reset",
                started_at=started_at,
                completed_at=utc_now(),
            )

    def status(self) -> dict[str, object]:
        return {
            "started": self._started,
            "busy": self._lock.locked(),
            "usage": {
                "requests": self.runtime.usage.requests,
                "input_tokens": self.runtime.usage.input_tokens,
                "output_tokens": self.runtime.usage.output_tokens,
                "total_tokens": self.runtime.usage.total_tokens,
                "cost_usd": self.runtime.usage.cost_usd,
                "context_tokens": self.runtime.usage.context_tokens,
            },
        }
