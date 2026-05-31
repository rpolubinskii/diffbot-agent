from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AgentConfig:
    backend: str
    model: str
    session_id: str
    session_db: str
    busy_policy: str


@dataclass(frozen=True)
class McpConfig:
    url: str


@dataclass(frozen=True)
class AudioConfig:
    host: str
    port: int
    voice_stream_enabled: bool


@dataclass(frozen=True)
class SecretsConfig:
    openai_api_key_env: str


@dataclass(frozen=True)
class AppConfig:
    agent: AgentConfig
    mcp: McpConfig
    audio: AudioConfig
    secrets: SecretsConfig


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}. Copy config.example.toml to config.toml.")

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    agent = _table(data, "agent")
    mcp = _table(data, "mcp")
    audio = _table(data, "audio")
    secrets = _table(data, "secrets")

    agent_config = AgentConfig(
        backend=_string(agent, "backend", "codex"),
        model=_string(agent, "model", "gpt-5.1"),
        session_id=_string(agent, "session_id", "diffbot-main"),
        session_db=_string(agent, "session_db", "diffbot-agent.sqlite3"),
        busy_policy=_string(agent, "busy_policy", "ignore"),
    )

    if agent_config.backend != "codex":
        raise ConfigError("V1 only supports [agent].backend = \"codex\".")
    if agent_config.busy_policy != "ignore":
        raise ConfigError("V1 only supports [agent].busy_policy = \"ignore\".")

    return AppConfig(
        agent=agent_config,
        mcp=McpConfig(url=_string(mcp, "url", "http://localhost:8080/mcp")),
        audio=AudioConfig(
            host=_string(audio, "host", "localhost"),
            port=_integer(audio, "port", 50051),
            voice_stream_enabled=_boolean(audio, "voice_stream_enabled", False),
        ),
        secrets=SecretsConfig(
            openai_api_key_env=_string(secrets, "openai_api_key_env", "OPENAI_API_KEY"),
        ),
    )


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table.")
    return value


def _string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string.")
    return value


def _integer(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer.")
    return value


def _boolean(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false.")
    return value
