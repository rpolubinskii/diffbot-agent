from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass

from diffbot_agent.agent_runtime import AgentRuntime
from diffbot_agent.audio_client import AudioCommandClient, stdin_commands
from diffbot_agent.config import AppConfig
from diffbot_agent.mcp_client import DiffbotMcpClient, compose_command_turn


@dataclass
class Orchestrator:
    config: AppConfig
    runtime: AgentRuntime
    mcp_client: DiffbotMcpClient
    audio_client: AudioCommandClient

    def __post_init__(self) -> None:
        self._turn_running = False

    async def run(self) -> None:
        await self.mcp_client.start()
        try:
            await self.runtime.start()
            try:
                async for command in self._commands():
                    await self._accept_command(command)
            finally:
                await self.runtime.stop()
        finally:
            await self.mcp_client.stop()

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
            await self._run_command_turn(command)
        except Exception as exc:
            print(f"Command turn failed: {exc}", file=sys.stderr)
        finally:
            self._turn_running = False

    async def _run_command_turn(self, command: str) -> None:
        robot_status = await self.mcp_client.read_robot_status()
        turn_text = await self.mcp_client.command_turn_prompt(command, robot_status)
        if turn_text is None:
            turn_text = compose_command_turn(command, robot_status)
        await self.runtime.run_turn(turn_text, robot_status)
