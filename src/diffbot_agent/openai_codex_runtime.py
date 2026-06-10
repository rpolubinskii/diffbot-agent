from __future__ import annotations

import inspect
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from diffbot_agent.command_memory import (
    CommandContextState,
    CommandMemoryStore,
    build_canonical_record,
    compose_command_input,
    sanitize_session_items,
    utc_now,
)
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
Use speak tool as the main way to communicate with the user. Use diffbot-mcp tools for robot state, navigation, vision, speech, and memory.
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
        self._model_provider = None
        self._command_memories: CommandMemoryStore | None = None

    async def start(self) -> None:
        profile = self.config.agent
        if profile.backend == "openai" and not profile.openai_api_key.strip():
            raise ConfigError(f"[agents.{profile.name}].openai_api_key is required for OpenAI.")
        if profile.backend == "ollama" and (
            not profile.model.strip() or not profile.base_url.strip()
        ):
            raise ConfigError(f"[agents.{profile.name}] requires model and base_url for Ollama.")

        from agents import Agent, SQLiteSession
        from agents import set_default_openai_key
        from agents.mcp import MCPServerStreamableHttp

        if profile.backend == "openai":
            set_default_openai_key(profile.openai_api_key)
            self._model_provider = None
        elif profile.backend == "ollama":
            from diffbot_agent.ollama_vision_provider import OllamaVisionProvider

            self._model_provider = OllamaVisionProvider(
                api_key=profile.api_key or "ollama",
                base_url=profile.base_url,
                use_responses=False,
            )
        else:
            raise ConfigError(f'Unsupported agent backend "{profile.backend}".')
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
            session_class = _image_sanitizing_session_class(SQLiteSession)
            self._session = session_class(
                profile.session_id,
                profile.session_db,
            )
            self._command_memories = CommandMemoryStore(profile.session_db)
            self._stack = stack
        except Exception:
            if self._command_memories is not None:
                await self._command_memories.close()
                self._command_memories = None
            if self._session is not None:
                close = getattr(self._session, "close", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                self._session = None
            await stack.aclose()
            raise

    async def run_turn(self, command: str, robot_status: str) -> None:
        if (
            self._agent is None
            or self._session is None
            or self._command_memories is None
        ):
            raise RuntimeError("OpenAI Agents runtime has not been started.")

        from agents import RunConfig, Runner

        recent_memories = await self._command_memories.latest(
            self.config.agent.session_id,
            self.config.agent_runtime.history_commands,
        )
        turn_text = compose_command_input(command, robot_status, recent_memories)
        command_state = CommandContextState(
            full_tool_rounds=self.config.agent_runtime.full_tool_rounds,
        )
        run_config_kwargs: dict[str, Any] = {
            "session_input_callback": _exclude_session_history,
            "call_model_input_filter": command_state.filter_model_input,
            "trace_include_sensitive_data": False,
        }
        if self._model_provider is not None:
            run_config_kwargs["model_provider"] = self._model_provider
        run_config = RunConfig(**run_config_kwargs)
        run_hooks = _build_run_hooks(self.config, command_state)
        started_at = utc_now()
        result = Runner.run_streamed(
            self._agent,
            turn_text,
            hooks=run_hooks,
            max_turns=self.config.agent_runtime.max_turns,
            run_config=run_config,
            session=self._session,
        )
        try:
            async for _event in result.stream_events():
                pass
        except Exception as exc:
            completed_at = utc_now()
            record = build_canonical_record(
                session_id=self.config.agent.session_id,
                started_at=started_at,
                completed_at=completed_at,
                command=command,
                completion_status=_completion_status(exc),
                items=_result_input_items(result),
                final_output=getattr(result, "final_output", None),
                error=exc,
            )
            try:
                await self._command_memories.add(record)
            except Exception as memory_exc:
                log_event(
                    "command.memory.error",
                    {
                        "command": command,
                        "error_type": type(memory_exc).__name__,
                        "error": str(memory_exc),
                    },
                )
            raise

        completed_at = utc_now()
        await self._command_memories.add(
            build_canonical_record(
                session_id=self.config.agent.session_id,
                started_at=started_at,
                completed_at=completed_at,
                command=command,
                completion_status="completed",
                items=_result_input_items(result),
                final_output=getattr(result, "final_output", None),
            )
        )

    async def stop(self) -> None:
        if self._session is not None:
            close = getattr(self._session, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
            self._session = None
        if self._command_memories is not None:
            await self._command_memories.close()
            self._command_memories = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._agent = None
            self._model_provider = None


OpenAICodexRuntime = OpenAIAgentsRuntime


def _build_run_hooks(
    config: AppConfig,
    command_state: CommandContextState,
) -> Any:
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
            command_state.mark_model_request_succeeded()
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


def _image_sanitizing_session_class(base_class: type[Any]) -> type[Any]:
    class ImageSanitizingSQLiteSession(base_class):  # type: ignore[misc, valid-type]
        async def add_items(self, items: list[Any]) -> None:
            await super().add_items(sanitize_session_items(items))

    return ImageSanitizingSQLiteSession


def _exclude_session_history(
    history_items: list[Any],
    new_items: list[Any],
) -> list[Any]:
    del history_items
    return new_items


def _result_input_items(result: Any) -> list[Any]:
    to_input_list = getattr(result, "to_input_list", None)
    if not callable(to_input_list):
        return []
    try:
        return list(to_input_list())
    except Exception:
        return []


def _completion_status(error: BaseException) -> str:
    if type(error).__name__ == "MaxTurnsExceeded":
        return "max_turns"
    return "failed"


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
