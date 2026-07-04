from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

from diffbot_agent.agent_runtime import AgentRuntime
from diffbot_agent.audio_client import AudioCommandClient
from diffbot_agent.config import AppConfig, ConfigError, load_config
from diffbot_agent.logging_utils import configure_logging
from diffbot_agent.openai_agents_runtime import OpenAIAgentsRuntime
from diffbot_agent.operator_input import OperatorInputCoordinator
from diffbot_agent.orchestrator import Orchestrator


def build_runtime(
    config: AppConfig,
    input_coordinator: OperatorInputCoordinator | None = None,
) -> AgentRuntime:
    if config.agent.backend in {"openai", "ollama"}:
        return OpenAIAgentsRuntime(
            config,
            elicitation_answer_provider=(
                input_coordinator.request_answer if input_coordinator is not None else None
            ),
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DiffBot agent orchestrator.")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Run the local A2A/MCP server.")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the local server.",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Port for the local server.",
    )
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config.toml.",
    )
    serve_parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear SDK conversation history before serving.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config.toml.",
    )
    parser.add_argument(
        "--reset-session",
        action="store_true",
        help="Clear SDK conversation history for the active profile.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        configure_logging(config.logging.level)
        if args.reset_session:
            asyncio.run(reset_session(config))
            print(
                f'Reset session "{config.agent.session_id}" in {config.agent.session_db}.'
            )
            if args.command != "serve":
                return

        if args.command == "serve":
            import uvicorn

            from diffbot_agent.server import create_app

            input_coordinator = OperatorInputCoordinator()
            runtime = build_runtime(config, input_coordinator)
            app = create_app(runtime, host=args.host, port=args.port)
            uvicorn.run(app, host=args.host, port=args.port)
            return

        input_coordinator = OperatorInputCoordinator()
        runtime = build_runtime(config, input_coordinator)
        audio_client = AudioCommandClient(config.audio)
        orchestrator = Orchestrator(
            config,
            runtime,
            audio_client,
            input_coordinator,
        )
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
