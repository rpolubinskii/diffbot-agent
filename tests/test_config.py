from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from diffbot_agent.config import ConfigError, load_config


_BASE = """
active_agent = "main"
[agent_runtime]
busy_policy = "ignore"
max_turns = 50
[agents.main]
backend = "openai"
model = "gpt-5.1"
openai_api_key = "sk-test"
session_id = "s"
session_db = "x.sqlite3"
[mcp]
url = "http://localhost:8080/mcp"
[audio]
host = "localhost"
port = 50052
voice_stream_enabled = false
reconnect_delay_seconds = 2.0
[logging]
level = "info"
"""


def _write(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "config.toml"
    path.write_text(text)
    return path


class ConfigTest(unittest.TestCase):
    def test_defaults_when_sections_absent(self) -> None:
        cfg = load_config(_write(_BASE))
        self.assertEqual(cfg.memory.backend, "sqlite")
        self.assertEqual(cfg.agent_runtime.compact_threshold, 240000)
        self.assertEqual(cfg.tool_categories, {})

    def test_memory_and_categories_parse(self) -> None:
        cfg = load_config(
            _write(
                _BASE
                + '\n[memory]\nbackend = "none"\n'
                + '[tool_categories]\npatrol = "navigation"\n'
            )
        )
        self.assertEqual(cfg.memory.backend, "none")
        self.assertEqual(cfg.tool_categories, {"patrol": "navigation"})

    def test_compact_threshold_override(self) -> None:
        cfg = load_config(_write(_BASE.replace("max_turns = 50", "max_turns = 50\ncompact_threshold = 90000")))
        self.assertEqual(cfg.agent_runtime.compact_threshold, 90000)

    def test_invalid_memory_backend_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(_write(_BASE + '\n[memory]\nbackend = "redis"\n'))

    def test_invalid_tool_category_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            load_config(_write(_BASE + '\n[tool_categories]\npatrol = "flying"\n'))


if __name__ == "__main__":
    unittest.main()
