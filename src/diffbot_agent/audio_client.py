from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import grpc
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.patch_stdout import patch_stdout

from diffbot_agent.config import AudioConfig
from diffbot_agent.logging_utils import LOGGER_NAME, log_event
from diffbot_agent.proto import audio_pb2
from diffbot_agent.session_usage import SessionUsage


_STREAM_VOICE_COMMANDS_METHOD = "/diffbot.audio.v1.AudioService/StreamVoiceCommands"


@dataclass(frozen=True)
class AudioCommandClient:
    config: AudioConfig

    async def command_stream(self) -> AsyncIterator[str]:
        if not self.config.voice_stream_enabled:
            return

        target = f"{self.config.host}:{self.config.port}"
        while True:
            channel = grpc.aio.insecure_channel(target)
            try:
                log_event("audio.command_stream.connecting", {"target": target})
                method = channel.unary_stream(
                    _STREAM_VOICE_COMMANDS_METHOD,
                    request_serializer=audio_pb2.StreamVoiceCommandsRequest.SerializeToString,
                    response_deserializer=audio_pb2.VoiceCommandEvent.FromString,
                )
                call = method(audio_pb2.StreamVoiceCommandsRequest())
                async for event in call:
                    if event.error:
                        log_event(
                            "audio.command_stream.event_error",
                            {"target": target, "error": event.error},
                            level=logging.WARNING,
                        )
                        continue

                    command = event.text.strip()
                    if command:
                        log_event(
                            "audio.command_stream.command",
                            {"target": target, "text": command},
                        )
                        yield command

                log_event(
                    "audio.command_stream.ended",
                    {"target": target},
                    level=logging.WARNING,
                )
            except asyncio.CancelledError:
                raise
            except grpc.RpcError as exc:
                log_event(
                    "audio.command_stream.rpc_error",
                    {
                        "target": target,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level=logging.WARNING,
                )
            finally:
                await channel.close()

            await asyncio.sleep(self.config.reconnect_delay_seconds)


SLASH_COMMANDS = ("/help", "/tokens", "/reset", "/quit")
_HELP_TEXT = (
    "Commands: /tokens (usage breakdown) · /reset (clear context + counters) · "
    "/help · /quit (or exit). ↑/↓ recall history. Anything else is sent to the agent."
)


def _build_prompt_session(history_path: Path | None) -> PromptSession[str]:
    history = FileHistory(str(history_path)) if history_path is not None else InMemoryHistory()
    completer = WordCompleter([*SLASH_COMMANDS, "exit", "quit"], sentence=True)
    return PromptSession(history=history, completer=completer)


@contextmanager
def _console_output() -> Iterator[None]:
    """Redraw async logs/prints above the live prompt instead of smearing it.

    ``patch_stdout`` swaps ``sys.stdout``/``sys.stderr`` for proxies that reprint
    the prompt after each write. The logging handler bound its stream at startup,
    so re-point it at the (now proxied) stderr for the duration of the session.
    """
    with patch_stdout():
        logger = logging.getLogger(LOGGER_NAME)
        restored = [
            (handler, handler.setStream(sys.stderr))
            for handler in logger.handlers
            if isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, logging.FileHandler)
        ]
        try:
            yield
        finally:
            for handler, previous in restored:
                if previous is not None:
                    handler.setStream(previous)


async def stdin_commands(
    usage: SessionUsage,
    *,
    history_path: Path | None = None,
    prompt: str = "diffbot> ",
) -> AsyncIterator[str]:
    session = _build_prompt_session(history_path)
    with _console_output():
        while True:
            try:
                line = await session.prompt_async(
                    prompt, bottom_toolbar=lambda: usage.toolbar_text()
                )
            except (EOFError, KeyboardInterrupt):
                break
            command = line.strip()
            lowered = command.lower()
            if lowered in {"exit", "quit", "/quit"}:
                break
            if lowered == "/help":
                print(_HELP_TEXT)
                continue
            if lowered == "/tokens":
                print(usage.breakdown())
                continue
            if command:
                yield command
