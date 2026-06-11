from __future__ import annotations

import asyncio
import io
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from diffbot_agent.command_memory import CommandContextState
from diffbot_agent.config import ConfigError, load_config
from diffbot_agent.logging_utils import (
    LOGGER_NAME,
    configure_logging,
    log_by_verbosity,
    log_event,
)
from diffbot_agent.openai_codex_runtime import (
    _build_run_hooks,
    _logging_mcp_server_class,
    _reasoning_texts,
)


@contextmanager
def _captured_logs(level: str):
    logger = logging.getLogger(LOGGER_NAME)
    old_handlers = logger.handlers[:]
    old_level = logger.level
    old_propagate = logger.propagate
    stream = io.StringIO()
    logger.handlers = [logging.StreamHandler(stream)]
    configure_logging(level)
    try:
        yield stream
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate


def _write_config(path: Path, logging_table: str = "") -> None:
    path.write_text(
        f'''active_agent = "main"

[agent_runtime]

[agents.main]
backend = "openai"
model = "gpt-test"
session_id = "session"
session_db = "session.sqlite3"
openai_api_key = "test-key"

[mcp]
url = "http://localhost:8080/mcp"

[audio]
voice_stream_enabled = false

{logging_table}
''',
        encoding="utf-8",
    )


def test_logging_level_defaults_to_info_and_accepts_debug(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path)
    assert load_config(config_path).logging.level == "info"

    _write_config(config_path, '[logging]\nlevel = "debug"')
    assert load_config(config_path).logging.level == "debug"


def test_logging_level_rejects_unknown_value(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, '[logging]\nlevel = "trace"')

    with pytest.raises(ConfigError, match='"info" or "debug"'):
        load_config(config_path)


def test_verbosity_router_selects_one_payload() -> None:
    with _captured_logs("info") as stream:
        log_by_verbosity(
            debug_event="mcp.tool.request",
            debug_payload={"tool": "nav_turn", "arguments": {"secret": 1}},
            info_event="mcp.tool.call",
            info_payload={"tool": "nav_turn"},
        )
        info_output = stream.getvalue()

    assert "mcp.tool.call" in info_output
    assert "nav_turn" in info_output
    assert "arguments" not in info_output
    assert "mcp.tool.request" not in info_output

    with _captured_logs("debug") as stream:
        log_by_verbosity(
            debug_event="mcp.tool.request",
            debug_payload={"tool": "nav_turn", "arguments": {"radians": 1}},
            info_event="mcp.tool.call",
            info_payload={"tool": "nav_turn"},
        )
        debug_output = stream.getvalue()

    assert "mcp.tool.request" in debug_output
    assert "arguments" in debug_output
    assert "mcp.tool.call" not in debug_output


def test_info_keeps_warnings_but_suppresses_diagnostics() -> None:
    with _captured_logs("info") as stream:
        log_event("llm.request", {"input": "hidden"})
        log_event("command.turn.error", {"error": "failed"}, level=logging.ERROR)
        output = stream.getvalue()

    assert "llm.request" not in output
    assert "command.turn.error" in output


def test_reasoning_extraction_preserves_order_and_deduplicates() -> None:
    response = {
        "output": [
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Plan route"},
                    {"type": "summary_text", "text": "Plan route"},
                ],
                "content": [
                    {"type": "reasoning_text", "text": "Check clearance"}
                ],
            },
            {"type": "message", "content": []},
        ]
    }

    assert _reasoning_texts(response) == ["Plan route", "Check clearance"]
    assert _reasoning_texts({"output": []}) == []


def test_info_runtime_logs_tool_name_and_reasoning_only() -> None:
    class FakeMcpBase:
        name = "diffbot-mcp"

        async def call_tool(self, tool_name, arguments, meta=None):
            return {"ok": True}

    async def exercise() -> str:
        config = SimpleNamespace(
            active_agent="main",
            agent=SimpleNamespace(backend="openai", model="gpt-test"),
        )
        state = CommandContextState(full_tool_rounds=1)
        hooks = _build_run_hooks(config, state)
        mcp_class = _logging_mcp_server_class(FakeMcpBase)

        with _captured_logs("info") as stream:
            await mcp_class().call_tool("nav_turn", {"radians": 1})
            await hooks.on_llm_start(None, SimpleNamespace(name="DiffBot"), "system", [])
            await hooks.on_llm_end(
                None,
                SimpleNamespace(name="DiffBot"),
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": "Turn safely"}
                            ],
                        }
                    ]
                },
            )
            return stream.getvalue()

    output = asyncio.run(exercise())

    assert "mcp.tool.call" in output
    assert '"tool":"nav_turn"' in output
    assert "llm.reasoning" in output
    assert "Turn safely" in output
    assert "radians" not in output
    assert "llm.request" not in output
    assert "llm.response" not in output


def test_legacy_config_uses_logging_level(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy.toml"
    config_path.write_text(
        '''[agent]
backend = "codex"

[secrets]
openai_api_key = "test-key"

[logging]
level = "debug"
''',
        encoding="utf-8",
    )

    assert load_config(config_path).logging.level == "debug"


def test_debug_runtime_retains_detailed_payloads() -> None:
    class FakeMcpBase:
        name = "diffbot-mcp"

        async def call_tool(self, tool_name, arguments, meta=None):
            return {"ok": True, "detail": "complete"}

    async def exercise() -> str:
        config = SimpleNamespace(
            active_agent="main",
            agent=SimpleNamespace(backend="openai", model="gpt-test"),
        )
        hooks = _build_run_hooks(config, CommandContextState(full_tool_rounds=1))
        mcp_class = _logging_mcp_server_class(FakeMcpBase)

        with _captured_logs("debug") as stream:
            await mcp_class().call_tool("nav_turn", {"radians": 1})
            await hooks.on_llm_start(
                None,
                SimpleNamespace(name="DiffBot"),
                "system prompt",
                [{"role": "user", "content": "turn"}],
            )
            await hooks.on_llm_end(
                None,
                SimpleNamespace(name="DiffBot"),
                {"output": [{"type": "message", "content": []}]},
            )
            return stream.getvalue()

    output = asyncio.run(exercise())

    assert "mcp.tool.request" in output
    assert '"arguments":{"radians":1}' in output
    assert "mcp.tool.response" in output
    assert "complete" in output
    assert "llm.request" in output
    assert "system prompt" in output
    assert "llm.response" in output
    assert "mcp.tool.call" not in output
    assert "llm.reasoning" not in output
