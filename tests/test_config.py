from __future__ import annotations

from pathlib import Path

import pytest

from diffbot_agent.config import ConfigError, load_config


def _write_config(path: Path, runtime: str = "") -> None:
    path.write_text(
        f"""
active_agent = "main"

[agent_runtime]
{runtime}

[agents.main]
backend = "openai"
model = "gpt-test"
session_id = "session"
session_db = "session.sqlite3"
openai_api_key = "test-key"

[mcp]
url = "http://localhost:8080/mcp"

[audio]
host = "localhost"
port = 50052
voice_stream_enabled = false
reconnect_delay_seconds = 1
""",
        encoding="utf-8",
    )


def test_runtime_context_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)

    config = load_config(config_path)

    assert config.agent_runtime.history_commands == 4
    assert config.agent_runtime.full_tool_rounds == 6


def test_runtime_context_allows_zero(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        "history_commands = 0\nfull_tool_rounds = 0",
    )

    config = load_config(config_path)

    assert config.agent_runtime.history_commands == 0
    assert config.agent_runtime.full_tool_rounds == 0


@pytest.mark.parametrize("key", ["history_commands", "full_tool_rounds"])
def test_runtime_context_rejects_negative_values(tmp_path: Path, key: str) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, f"{key} = -1")

    with pytest.raises(ConfigError, match="greater than or equal to zero"):
        load_config(config_path)
