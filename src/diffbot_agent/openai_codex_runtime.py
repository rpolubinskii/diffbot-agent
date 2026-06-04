from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from diffbot_agent.config import AppConfig, ConfigError
from diffbot_agent.logging_utils import (
    elapsed_ms,
    has_error_marker,
    log_event,
    monotonic_ms,
    serialize_for_json,
)


INSTRUCTIONS = """You are a differential long-running robot control agent.

You receive one operator or voice command per turn. Each turn includes fresh robot://status.
Use speak tool as the main way to communicate with the user. Use diffbot-mcp tools for robot state, navigation, vision, speech, and memory. When motion is uncertain, unsafe, interrupted, or failing, stop or cancel motion.
"""

MCP_CLIENT_SESSION_TIMEOUT_SECONDS = 90
MCP_MAX_RETRY_ATTEMPTS = 0


@dataclass
class OpenAIAgentsRuntime:
    config: AppConfig

    def __post_init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._agent = None
        self._session = None
        self._run_config = None
        self._run_hooks = None

    async def start(self) -> None:
        profile = self.config.agent
        if profile.backend == "openai" and not profile.openai_api_key.strip():
            raise ConfigError(f"[agents.{profile.name}].openai_api_key is required for OpenAI.")
        if profile.backend == "ollama" and (
            not profile.model.strip() or not profile.base_url.strip()
        ):
            raise ConfigError(f"[agents.{profile.name}] requires model and base_url for Ollama.")

        from agents import Agent, RunConfig, SQLiteSession
        from agents import set_default_openai_key
        from agents.mcp import MCPServerStreamableHttp

        if profile.backend == "openai":
            set_default_openai_key(profile.openai_api_key)
            self._run_config = None
        elif profile.backend == "ollama":
            from diffbot_agent.ollama_vision_provider import OllamaVisionProvider

            self._run_config = RunConfig(
                model_provider=OllamaVisionProvider(
                    api_key=profile.api_key or "ollama",
                    base_url=profile.base_url,
                    use_responses=False,
                )
            )
        else:
            raise ConfigError(f'Unsupported agent backend "{profile.backend}".')
        self._run_hooks = _build_run_hooks(self.config)
        mcp_server_class = _logging_mcp_server_class(MCPServerStreamableHttp)

        stack = AsyncExitStack()
        try:
            mcp_server = await stack.enter_async_context(
                mcp_server_class(
                    name="diffbot-mcp",
                    params={
                        "url": self.config.mcp.url,
                        "timeout": 10,
                        "sse_read_timeout": 300,
                    },
                    cache_tools_list=True,
                    client_session_timeout_seconds=MCP_CLIENT_SESSION_TIMEOUT_SECONDS,
                    max_retry_attempts=MCP_MAX_RETRY_ATTEMPTS,
                )
            )

            self._agent = Agent(
                name="DiffBot",
                instructions=INSTRUCTIONS,
                model=profile.model,
                mcp_servers=[mcp_server],
                mcp_config={
                    "convert_schemas_to_strict": True,
                    "include_server_in_tool_names": True,
                },
            )
            self._session = SQLiteSession(
                profile.session_id,
                profile.session_db,
            )
            self._stack = stack
        except Exception:
            await stack.aclose()
            raise

    async def run_turn(self, user_text: str, robot_status: str) -> None:
        if self._agent is None or self._session is None:
            raise RuntimeError("OpenAI Agents runtime has not been started.")

        from agents import Runner

        turn_text = _ensure_robot_status(user_text, robot_status)
        result = Runner.run_streamed(
            self._agent,
            turn_text,
            hooks=self._run_hooks,
            max_turns=self.config.agent_runtime.max_turns,
            run_config=self._run_config,
            session=self._session,
        )

        async for _event in result.stream_events():
            pass

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._agent = None
            self._session = None
            self._run_config = None
            self._run_hooks = None


OpenAICodexRuntime = OpenAIAgentsRuntime


def _build_run_hooks(config: AppConfig) -> Any:
    from agents.lifecycle import RunHooksBase

    class LoggingRunHooks(RunHooksBase):
        async def on_llm_start(
            self,
            context: Any,
            agent: Any,
            system_prompt: str | None,
            input_items: list[Any],
        ) -> None:
            log_event(
                "llm.request",
                {
                    "active_agent": config.active_agent,
                    "backend": config.agent.backend,
                    "model": config.agent.model,
                    "agent": getattr(agent, "name", None),
                    "system_prompt": system_prompt,
                    "input_items": input_items,
                },
            )

        async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
            log_event(
                "llm.response",
                {
                    "active_agent": config.active_agent,
                    "backend": config.agent.backend,
                    "model": config.agent.model,
                    "agent": getattr(agent, "name", None),
                    "response": serialize_for_json(response),
                },
            )

    return LoggingRunHooks()


def _logging_mcp_server_class(base_class: type[Any]) -> type[Any]:
    class LoggingMCPServerStreamableHttp(base_class):  # type: ignore[misc, valid-type]
        async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
            started = monotonic_ms()
            log_event(
                "mcp.list_tools.request",
                {"server": self.name},
            )
            try:
                result = await super().list_tools(*args, **kwargs)
                log_event(
                    "mcp.list_tools.response",
                    {
                        "server": self.name,
                        "duration_ms": elapsed_ms(started),
                        "tools": serialize_for_json(result),
                    },
                )
                return result
            except Exception as exc:
                log_event(
                    "mcp.list_tools.error",
                    {
                        "server": self.name,
                        "duration_ms": elapsed_ms(started),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

        async def call_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any] | None,
            meta: dict[str, Any] | None = None,
        ) -> Any:
            started = monotonic_ms()
            log_event(
                "mcp.tool.request",
                {
                    "server": self.name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "meta": meta,
                },
            )
            try:
                result = await super().call_tool(tool_name, arguments, meta=meta)
                serialized_result = serialize_for_json(result)
                log_event(
                    "mcp.tool.response",
                    {
                        "server": self.name,
                        "tool": tool_name,
                        "duration_ms": elapsed_ms(started),
                        "result": serialized_result,
                    },
                )
                if has_error_marker(serialized_result):
                    log_event(
                        "mcp.tool.result_error",
                        {
                            "server": self.name,
                            "tool": tool_name,
                            "duration_ms": elapsed_ms(started),
                            "result": serialized_result,
                        },
                    )
                return result
            except Exception as exc:
                log_event(
                    "mcp.tool.error",
                    {
                        "server": self.name,
                        "tool": tool_name,
                        "duration_ms": elapsed_ms(started),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                raise

    return LoggingMCPServerStreamableHttp


def _ensure_robot_status(user_text: str, robot_status: str) -> str:
    if "Robot status:" in user_text or "robot://status" in user_text:
        return user_text
    return f"{user_text}\n\nFresh robot://status:\n{robot_status}"
