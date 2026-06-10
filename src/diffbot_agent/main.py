from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

from diffbot_agent.agent_runtime import AgentRuntime
from diffbot_agent.audio_client import AudioCommandClient
from diffbot_agent.command_memory import clear_command_memories
from diffbot_agent.config import AppConfig, ConfigError, load_config
from diffbot_agent.logging_utils import configure_logging
from diffbot_agent.mcp_client import DiffbotMcpClient
from diffbot_agent.openai_codex_runtime import OpenAIAgentsRuntime
from diffbot_agent.orchestrator import Orchestrator


def build_runtime(config: AppConfig) -> AgentRuntime:
    if config.agent.backend in {"openai", "ollama"}:
        return OpenAIAgentsRuntime(config)
    raise ConfigError(f'Unsupported agent backend "{config.agent.backend}".')


async def reset_session(config: AppConfig) -> None:
    from agents import SQLiteSession

    session = SQLiteSession(config.agent.session_id, config.agent.session_db)
    try:
        result = session.clear_session()
        if inspect.isawaitable(result):
            await result
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        clear_command_memories(
            config.agent.session_db,
            config.agent.session_id,
        )


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run the DiffBot agent orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config.toml.",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear SDK history and canonical command memory for the active profile.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        if args.reset_session:
            asyncio.run(reset_session(config))
            print(
                f'Reset session "{config.agent.session_id}" in {config.agent.session_db}.'
            )
            return

        runtime = build_runtime(config)
        mcp_client = DiffbotMcpClient(config.mcp.url)
        audio_client = AudioCommandClient(config.audio)
        orchestrator = Orchestrator(config, runtime, mcp_client, audio_client)
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        pass
    except ConfigError as exc:
        print(f"Invalid config: {exc}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"diffbot-agent failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
