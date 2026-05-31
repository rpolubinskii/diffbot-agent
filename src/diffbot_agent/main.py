from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from diffbot_agent.audio_client import AudioCommandClient
from diffbot_agent.config import ConfigError, load_config
from diffbot_agent.mcp_client import DiffbotMcpClient
from diffbot_agent.openai_codex_runtime import OpenAICodexRuntime
from diffbot_agent.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DiffBot agent orchestrator.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="Path to config.toml.",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        runtime = OpenAICodexRuntime(config)
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
