from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AgentProfileConfig:
    name: str
    backend: str
    model: str
    session_id: str
    session_db: str
    openai_api_key: str = ""
    base_url: str = ""
    api_key: str = ""


@dataclass(frozen=True)
class AgentRuntimeConfig:
    busy_policy: str
    max_turns: int
    history_commands: int = 4
    full_tool_rounds: int = 6
    compact_threshold: int = 240000


@dataclass(frozen=True)
class MemoryConfig:
    backend: str = "sqlite"


@dataclass(frozen=True)
class McpConfig:
    url: str


@dataclass(frozen=True)
class AudioConfig:
    host: str
    port: int
    voice_stream_enabled: bool
    reconnect_delay_seconds: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class AppConfig:
    active_agent: str
    agent: AgentProfileConfig
    agents: dict[str, AgentProfileConfig]
    agent_runtime: AgentRuntimeConfig
    memory: MemoryConfig
    tool_categories: dict[str, str]
    mcp: McpConfig
    audio: AudioConfig
    logging: LoggingConfig


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}. Copy config.example.toml to config.toml.")

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc

    mcp = _table(data, "mcp")
    audio = _table(data, "audio")
    logging_config = _table(data, "logging")
    memory = _table(data, "memory")
    active_agent, agents, runtime_config = _load_agent_config(data)

    return AppConfig(
        active_agent=active_agent,
        agent=agents[active_agent],
        agents=agents,
        agent_runtime=runtime_config,
        memory=MemoryConfig(backend=_memory_backend(memory)),
        tool_categories=_tool_categories(_table(data, "tool_categories")),
        mcp=McpConfig(url=_string(mcp, "url", "http://localhost:8080/mcp")),
        audio=AudioConfig(
            host=_string(audio, "host", "localhost"),
            port=_integer(audio, "port", 50052),
            voice_stream_enabled=_boolean(audio, "voice_stream_enabled", False),
            reconnect_delay_seconds=_number(audio, "reconnect_delay_seconds", 2.0),
        ),
        logging=LoggingConfig(level=_logging_level(logging_config)),
    )


def _load_agent_config(
    data: dict[str, Any],
) -> tuple[str, dict[str, AgentProfileConfig], AgentRuntimeConfig]:
    if "agents" in data or "active_agent" in data:
        active_agent = _string(data, "active_agent", "")
        agents_table = _table(data, "agents")
        if not agents_table:
            raise ConfigError("[agents] must contain at least one agent profile.")

        agents = {
            name: _profile_from_table(name, _profile_table(value, name))
            for name, value in agents_table.items()
        }
        if active_agent not in agents:
            raise ConfigError(f'active_agent "{active_agent}" is not configured under [agents].')

        _validate_selected_profile(agents[active_agent])
        return active_agent, agents, _runtime_config(_table(data, "agent_runtime"))

    return _legacy_agent_config(data)


def _legacy_agent_config(
    data: dict[str, Any],
) -> tuple[str, dict[str, AgentProfileConfig], AgentRuntimeConfig]:
    agent = _table(data, "agent")
    secrets = _table(data, "secrets")
    backend = _string(agent, "backend", "openai")

    profile = AgentProfileConfig(
        name="legacy-agent",
        backend=backend,
        model=_string(agent, "model", "gpt-5.1"),
        session_id=_string(agent, "session_id", "diffbot-main"),
        session_db=_string(agent, "session_db", "diffbot-agent.sqlite3"),
        openai_api_key=_optional_string(secrets, "openai_api_key", ""),
    )
    _validate_selected_profile(profile)

    runtime_config = AgentRuntimeConfig(
        busy_policy=_string(agent, "busy_policy", "ignore"),
        max_turns=_positive_integer(agent, "max_turns", 50),
        history_commands=_non_negative_integer(agent, "history_commands", 4),
        full_tool_rounds=_non_negative_integer(agent, "full_tool_rounds", 6),
        compact_threshold=_positive_integer(agent, "compact_threshold", 240000),
    )
    _validate_runtime_config(runtime_config)
    return profile.name, {profile.name: profile}, runtime_config


def _profile_from_table(name: str, data: dict[str, Any]) -> AgentProfileConfig:
    backend = _string(data, "backend", "")
    if backend not in {"openai", "ollama"}:
        raise ConfigError(f'[agents.{name}].backend must be "openai" or "ollama".')

    return AgentProfileConfig(
        name=name,
        backend=backend,
        model=_optional_string(data, "model", ""),
        session_id=_string(data, "session_id", "diffbot-main"),
        session_db=_string(data, "session_db", "diffbot-agent.sqlite3"),
        openai_api_key=_optional_string(data, "openai_api_key", ""),
        base_url=_optional_string(data, "base_url", ""),
        api_key=_optional_string(data, "api_key", "ollama"),
    )


def _profile_table(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"[agents.{name}] must be a TOML table.")
    return value


def _runtime_config(data: dict[str, Any]) -> AgentRuntimeConfig:
    config = AgentRuntimeConfig(
        busy_policy=_string(data, "busy_policy", "ignore"),
        max_turns=_positive_integer(data, "max_turns", 50),
        history_commands=_non_negative_integer(data, "history_commands", 4),
        full_tool_rounds=_non_negative_integer(data, "full_tool_rounds", 6),
        compact_threshold=_positive_integer(data, "compact_threshold", 240000),
    )
    _validate_runtime_config(config)
    return config


def _validate_runtime_config(config: AgentRuntimeConfig) -> None:
    if config.busy_policy != "ignore":
        raise ConfigError('V1 only supports busy_policy = "ignore".')


def _validate_selected_profile(profile: AgentProfileConfig) -> None:
    if profile.backend == "openai":
        if not profile.model.strip():
            raise ConfigError(f"[agents.{profile.name}].model is required for OpenAI.")
        if not profile.openai_api_key.strip():
            raise ConfigError(f"[agents.{profile.name}].openai_api_key is required for OpenAI.")
        return

    if profile.backend == "ollama":
        if not profile.model.strip():
            raise ConfigError(f"[agents.{profile.name}].model is required for Ollama.")
        if not profile.base_url.strip():
            raise ConfigError(f"[agents.{profile.name}].base_url is required for Ollama.")
        return

    raise ConfigError(f'[agents.{profile.name}].backend must be "openai" or "ollama".')


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


def _optional_string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string.")
    return value


def _integer(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{key} must be an integer.")
    return value


def _positive_integer(data: dict[str, Any], key: str, default: int) -> int:
    value = _integer(data, key, default)
    if value <= 0:
        raise ConfigError(f"{key} must be greater than zero.")
    return value


def _non_negative_integer(data: dict[str, Any], key: str, default: int) -> int:
    value = _integer(data, key, default)
    if value < 0:
        raise ConfigError(f"{key} must be greater than or equal to zero.")
    return value


def _number(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{key} must be a number.")
    if value < 0:
        raise ConfigError(f"{key} must be greater than or equal to zero.")
    return float(value)


def _boolean(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false.")
    return value


def _logging_level(data: dict[str, Any]) -> str:
    level = _string(data, "level", "info").lower()
    if level not in {"info", "debug"}:
        raise ConfigError('logging level must be "info" or "debug".')
    return level


def _memory_backend(data: dict[str, Any]) -> str:
    backend = _string(data, "backend", "sqlite").lower()
    if backend not in {"sqlite", "none", "diffbot_memory"}:
        raise ConfigError('[memory].backend must be "sqlite", "none", or "diffbot_memory".')
    return backend


def _tool_categories(data: dict[str, Any]) -> dict[str, str]:
    from diffbot_agent.episode import TOOL_CATEGORIES

    categories: dict[str, str] = {}
    for name, value in data.items():
        if not isinstance(value, str) or value not in TOOL_CATEGORIES:
            allowed = ", ".join(sorted(TOOL_CATEGORIES))
            raise ConfigError(
                f'[tool_categories].{name} must be one of: {allowed}.'
            )
        categories[name] = value
    return categories
