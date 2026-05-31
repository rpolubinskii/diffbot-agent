from __future__ import annotations

from typing import Protocol


class AgentRuntime(Protocol):
    async def start(self) -> None:
        """Initialize the runtime and its long-lived session."""

    async def run_turn(self, user_text: str, robot_status: str) -> None:
        """Run one user command turn against the existing session."""

    async def stop(self) -> None:
        """Release runtime resources."""
