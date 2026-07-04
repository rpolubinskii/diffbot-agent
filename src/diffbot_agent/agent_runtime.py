from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from diffbot_agent.session_usage import SessionUsage


TurnStatus = Literal["completed", "failed", "busy", "timeout"]


@dataclass(frozen=True)
class TurnResult:
    status: TurnStatus
    final_text: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    tool_events: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def compact_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"status": self.status}
        if self.final_text:
            data["final_text"] = self.final_text
        if self.started_at:
            data["started_at"] = self.started_at
        if self.completed_at:
            data["completed_at"] = self.completed_at
        if self.error:
            data["error"] = self.error
        if self.tool_events:
            data["tool_events"] = list(self.tool_events)
        return data


class AgentRuntime(Protocol):
    usage: SessionUsage
    """Live token/cost totals for the session, updated as turns run."""

    async def start(self) -> None:
        """Initialize the runtime and its long-lived session."""

    async def run_turn(self, command: str) -> TurnResult:
        """Run one user command turn against the existing session."""

    async def reset(self) -> None:
        """Clear conversation history and cross-command memory in place."""

    async def stop(self) -> None:
        """Release runtime resources."""
