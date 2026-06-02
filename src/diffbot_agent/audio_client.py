from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass

import grpc

from diffbot_agent.config import AudioConfig
from diffbot_agent.logging_utils import log_event
from diffbot_agent.proto import audio_pb2


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


async def stdin_commands(prompt: str = "diffbot> ") -> AsyncIterator[str]:
    while True:
        try:
            line = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            break
        command = line.strip()
        if command.lower() in {"exit", "quit"}:
            break
        if command:
            yield command
