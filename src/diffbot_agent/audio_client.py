from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from diffbot_agent.config import AudioConfig


@dataclass(frozen=True)
class AudioCommandClient:
    config: AudioConfig

    async def command_stream(self) -> AsyncIterator[str]:
        if not self.config.voice_stream_enabled:
            return
        raise NotImplementedError(
            "diffbot-audio does not expose a VTT command stream yet. "
            "Set [audio].voice_stream_enabled = false to use stdin."
        )
        yield ""


async def stdin_commands(prompt: str = "diffbot> ") -> AsyncIterator[str]:
    while True:
        try:
            line = await asyncio.to_thread(input, prompt)
        except EOFError:
            break
        command = line.strip()
        if command:
            yield command
