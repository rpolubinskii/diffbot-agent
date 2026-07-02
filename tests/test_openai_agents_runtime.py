from __future__ import annotations

from diffbot_agent.config import AgentProfileConfig, AgentRuntimeConfig
from diffbot_agent.openai_agents_runtime import (
    OLLAMA_REASONING_EFFORT,
    _model_settings_for_profile,
)


def _runtime_config(compact_threshold: int = 240000) -> AgentRuntimeConfig:
    return AgentRuntimeConfig(
        busy_policy="ignore",
        max_turns=50,
        compact_threshold=compact_threshold,
    )


def test_ollama_model_settings_pin_reasoning_effort_to_max() -> None:
    profile = AgentProfileConfig(
        name="local-ollama",
        backend="ollama",
        model="gemma4-26b-it-a4b",
        base_url="http://localhost:11434/v1",
        session_id="diffbot-ollama",
        session_db="diffbot-agent.sqlite3",
    )

    settings = _model_settings_for_profile(profile, _runtime_config())

    assert settings.extra_body == {"reasoning_effort": OLLAMA_REASONING_EFFORT}
    assert settings.reasoning is None


def test_openai_model_settings_keep_context_compaction() -> None:
    profile = AgentProfileConfig(
        name="openai-main",
        backend="openai",
        model="gpt-5.1",
        session_id="diffbot-main",
        session_db="diffbot-agent.sqlite3",
        openai_api_key="test-key",
    )

    settings = _model_settings_for_profile(profile, _runtime_config(compact_threshold=12345))

    assert settings.context_management == [
        {"type": "compaction", "compact_threshold": 12345}
    ]
    assert settings.extra_body is None
