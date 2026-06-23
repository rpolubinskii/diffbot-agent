from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

from diffbot_agent.agent_runtime import AgentRuntime
from diffbot_agent.audio_client import AudioCommandClient, stdin_commands
from diffbot_agent.config import AppConfig
from diffbot_agent.logging_utils import log_event
from diffbot_agent.mcp_client import DiffbotMcpClient
from diffbot_agent.operator_input import OperatorInputCoordinator, OperatorInputRoute


@dataclass
class Orchestrator:
    config: AppConfig
    runtime: AgentRuntime
    mcp_client: DiffbotMcpClient
    audio_client: AudioCommandClient
    input_coordinator: OperatorInputCoordinator | None = None

    def __post_init__(self) -> None:
        self._turn_running = False
        if self.input_coordinator is None:
            self.input_coordinator = OperatorInputCoordinator()

    async def run(self) -> None:
        await self.mcp_client.start()
        try:
            await self.runtime.start()
            producer = asyncio.create_task(self._produce_inputs())
            try:
                while self.input_coordinator is not None:
                    command = await self.input_coordinator.next_command()
                    if command is None:
                        break
                    await self._accept_command(command)
            finally:
                producer.cancel()
                with suppress(asyncio.CancelledError):
                    await producer
                await self.runtime.stop()
        finally:
            await self.mcp_client.stop()

    async def _produce_inputs(self) -> None:
        assert self.input_coordinator is not None
        try:
            async for command in self._commands():
                route = await self.input_coordinator.submit_answer(command)
                if route is OperatorInputRoute.ELICITATION:
                    continue
                if route is OperatorInputRoute.CLOSED:
                    break
                if self._turn_running and self.config.agent_runtime.busy_policy == "ignore":
                    print(f"Ignoring command while agent turn is running: {command}", file=sys.stderr)
                    continue
                await self.input_coordinator.enqueue_command(command)
        finally:
            await self.input_coordinator.close()

    async def _commands(self) -> AsyncIterator[str]:
        if self.config.audio.voice_stream_enabled:
            async for command in self.audio_client.command_stream():
                yield command
        else:
            print("Voice command stream disabled; using manual stdin commands.", file=sys.stderr)
            async for command in stdin_commands():
                yield command

    async def _accept_command(self, command: str) -> None:
        if self._turn_running:
            if self.config.agent_runtime.busy_policy == "ignore":
                print(f"Ignoring command while agent turn is running: {command}", file=sys.stderr)
                return

        self._turn_running = True
        try:
            if _is_reset_command(command):
                await self._reset_context()
            else:
                await self._run_command_turn(command)
        except Exception as exc:
            log_event(
                "command.turn.error",
                {
                    "command": command,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                level=logging.ERROR,
            )
            print(f"Command turn failed: {exc}", file=sys.stderr)
        finally:
            self._turn_running = False

    async def _run_command_turn(self, command: str) -> None:
        robot_status = await self.mcp_client.read_robot_status()
        await self.runtime.run_turn(command, robot_status)

    async def _reset_context(self) -> None:
        await self.runtime.reset()
        log_event("command.reset", {})
        print("Context reset: cleared conversation history and memory.", file=sys.stderr)


_RESET_COMMANDS = frozenset(
    {
        "reset context",
        "reset the context",
        "reset memory",
        "reset your memory",
        "clear context",
        "clear the context",
        "clear memory",
        "clear your memory",
        "reset session",
        "clear session",
        "forget everything",
    }
)


def _is_reset_command(command: str) -> bool:
    normalized = " ".join(command.lower().split()).strip(" .!?,")
    return normalized in _RESET_COMMANDS
