from __future__ import annotations

import asyncio
import inspect
import logging
import time
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

from datetime import datetime, timedelta

from diffbot_agent.context_window import CommandContextState, compose_command_input
from diffbot_agent.elicitation import ElicitationAnswerProvider, build_elicitation_callback
from diffbot_agent.episode import (
    build_canonical_record,
    set_mcp_tool_categories,
    set_tool_category_overrides,
    utc_now,
)
from diffbot_agent.memory_backend import (
    DiffbotMcpMemoryBackend,
    MemoryBackend,
    NullMemoryBackend,
    SqliteRecencyBackend,
)
from diffbot_agent.mcp_logging import McpRunLogger
from diffbot_agent.sanitize import sanitize_session_items
from diffbot_agent.config import AppConfig, ConfigError
from diffbot_agent.logging_utils import (
    log_event,
    log_by_verbosity,
    serialize_for_json,
)


INSTRUCTIONS = """You are a differential long-running robot control agent.

Use speak tool as the main way to communicate with the user. Use diffbot-mcp tools for robot state, navigation, vision, speech, and memory.
"""

MCP_CLIENT_SESSION_TIMEOUT_SECONDS = 90
MCP_MAX_RETRY_ATTEMPTS = 0

# Contract with diffbot-mcp ToolCategoryMeta.CATEGORY_KEY — keep in sync.
TOOL_CATEGORY_META_KEY = "diffbot.dev/category"


@dataclass
class OpenAIAgentsRuntime:
    config: AppConfig
    elicitation_answer_provider: ElicitationAnswerProvider | None = None

    def __post_init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._agent = None
        self._session = None
        self._model_provider = None
        self._memory: MemoryBackend | None = None
        self._compact_locally = True

    async def start(self) -> None:
        profile = self.config.agent
        set_tool_category_overrides(self.config.tool_categories)
        if profile.backend == "openai" and not profile.openai_api_key.strip():
            raise ConfigError(f"[agents.{profile.name}].openai_api_key is required for OpenAI.")
        if profile.backend == "ollama" and (
            not profile.model.strip() or not profile.base_url.strip()
        ):
            raise ConfigError(f"[agents.{profile.name}] requires model and base_url for Ollama.")

        from agents import Agent, SQLiteSession
        from agents import set_default_openai_key
        from agents.mcp import MCPServerStreamableHttp
        from agents.model_settings import ModelSettings

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

        # Ollama has no server-side compaction; it compacts tool rounds locally.
        self._compact_locally = profile.backend != "openai"
        model_settings = ModelSettings()
        if profile.backend == "openai" and self.config.agent_runtime.compact_threshold > 0:
            model_settings = ModelSettings(
                context_management=[
                    {
                        "type": "compaction",
                        "compact_threshold": self.config.agent_runtime.compact_threshold,
                    }
                ]
            )
        elicitation_callback = build_elicitation_callback(
            self.elicitation_answer_provider or _cancel_elicitation
        )
        mcp_server_class, mcp_server_extra_kwargs = _mcp_server_class_and_kwargs(
            MCPServerStreamableHttp,
            elicitation_callback,
        )

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
                    **mcp_server_extra_kwargs,
                )
            )

            self._agent = Agent(
                name="DiffBot",
                instructions=INSTRUCTIONS,
                model=profile.model,
                model_settings=model_settings,
                mcp_servers=[mcp_server],
                mcp_config={
                    "convert_schemas_to_strict": True,
                    "include_server_in_tool_names": True,
                },
            )
            await _load_mcp_tool_categories(mcp_server)
            session_class = _image_sanitizing_session_class(SQLiteSession)
            self._session = session_class(
                profile.session_id,
                profile.session_db,
            )
            self._memory = _build_memory_backend(self.config)
            await self._memory.start()
            self._stack = stack
        except Exception:
            if self._memory is not None:
                await self._memory.close()
                self._memory = None
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
            or self._memory is None
        ):
            raise RuntimeError("OpenAI Agents runtime has not been started.")

        from agents import RunConfig, Runner

        started_at = utc_now()
        started_monotonic = time.monotonic()
        historical_memory = await self._memory.recall(
            query=command,
            limit=self.config.agent_runtime.history_commands,
            now=datetime.fromisoformat(started_at),
        )
        turn_text = compose_command_input(
            command,
            robot_status,
            historical_memory,
            started_at=started_at,
        )
        command_state = CommandContextState(
            full_tool_rounds=self.config.agent_runtime.full_tool_rounds,
            command=command,
            robot_status=robot_status,
            historical_memory=historical_memory,
            compact_locally=self._compact_locally,
            started_at=started_at,
            started_monotonic=started_monotonic,
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
        mcp_logger = McpRunLogger()
        result = Runner.run_streamed(
            self._agent,
            turn_text,
            hooks=run_hooks,
            max_turns=self.config.agent_runtime.max_turns,
            run_config=run_config,
            session=self._session,
        )
        try:
            async for event in result.stream_events():
                mcp_logger.handle_event(event)
        except Exception as exc:
            mcp_logger.log_pending_errors(exc)
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
                await self._memory.add_episode(record)
            except Exception as memory_exc:
                log_event(
                    "command.memory.error",
                    {
                        "command": command,
                        "error_type": type(memory_exc).__name__,
                        "error": str(memory_exc),
                    },
                    level=logging.ERROR,
                )
            raise

        completed_at = utc_now()
        await self._memory.add_episode(
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

    async def reset(self) -> None:
        if self._session is not None:
            clear = getattr(self._session, "clear_session", None)
            if callable(clear):
                result = clear()
                if inspect.isawaitable(result):
                    await result
        if self._memory is not None:
            await self._memory.reset()

    async def stop(self) -> None:
        if self._session is not None:
            close = getattr(self._session, "close", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
            self._session = None
        if self._memory is not None:
            await self._memory.close()
            self._memory = None
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._agent = None
            self._model_provider = None


def _build_memory_backend(config: AppConfig) -> MemoryBackend:
    if config.memory.backend == "diffbot_memory":
        return DiffbotMcpMemoryBackend(config.mcp.url)
    if config.memory.backend == "sqlite":
        return SqliteRecencyBackend(config.agent.session_db, config.agent.session_id)
    return NullMemoryBackend()


async def _cancel_elicitation(timeout_seconds: float) -> str | None:
    del timeout_seconds
    return None


async def _load_mcp_tool_categories(mcp_server: Any) -> None:
    """Read tool categories from MCP ``_meta``; non-fatal (the heuristic covers gaps)."""
    try:
        tools = await mcp_server.list_tools()
    except Exception as exc:
        log_event(
            "mcp.tool_categories.error",
            {"error_type": type(exc).__name__, "error": str(exc)},
            level=logging.WARNING,
        )
        return
    categories: dict[str, str] = {}
    for tool in tools or []:
        name = getattr(tool, "name", None)
        meta = getattr(tool, "meta", None)
        if isinstance(name, str) and isinstance(meta, dict):
            category = meta.get(TOOL_CATEGORY_META_KEY)
            if isinstance(category, str) and category:
                categories[name] = category
    set_mcp_tool_categories(categories)
    log_event("mcp.tool_categories.loaded", {"count": len(categories)})


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
            reasoning = _reasoning_texts(response)
            log_by_verbosity(
                debug_event="llm.response",
                debug_payload={
                    "active_agent": config.active_agent,
                    "backend": config.agent.backend,
                    "model": config.agent.model,
                    "agent": getattr(agent, "name", None),
                    "response": serialize_for_json(response),
                },
                info_event="llm.reasoning" if reasoning else None,
                info_payload={"text": reasoning},
            )

    return LoggingRunHooks()


def _reasoning_texts(response: Any) -> list[str]:
    serialized = serialize_for_json(response)
    if not isinstance(serialized, dict):
        return []

    output = serialized.get("output")
    if not isinstance(output, list):
        return []

    texts: list[str] = []
    seen: set[str] = set()
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for key in ("summary", "content"):
            parts = item.get(key)
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text = text.strip()
                    if text and text not in seen:
                        seen.add(text)
                        texts.append(text)
    return texts


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


def _mcp_server_class_and_kwargs(
    base_class: type[Any],
    elicitation_callback: Any,
) -> tuple[type[Any], dict[str, Any]]:
    if "elicitation_callback" in inspect.signature(base_class).parameters:
        return base_class, {"elicitation_callback": elicitation_callback}
    return _elicitation_mcp_server_class(base_class), {
        "elicitation_callback": elicitation_callback,
    }


def _elicitation_mcp_server_class(base_class: type[Any]) -> type[Any]:
    class ElicitationMCPServerStreamableHttp(base_class):  # type: ignore[misc, valid-type]
        def __init__(
            self,
            *args: Any,
            elicitation_callback: Any | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.elicitation_callback = elicitation_callback

        async def connect(self) -> None:
            from agents.exceptions import UserError
            from agents.mcp.server import ClientSession, httpx, logger

            connection_succeeded = False
            try:
                transport = await self.exit_stack.enter_async_context(self.create_streams())
                read, write, *rest = transport
                self._get_session_id = rest[0] if rest and callable(rest[0]) else None

                session = await self.exit_stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        timedelta(seconds=self.client_session_timeout_seconds)
                        if self.client_session_timeout_seconds
                        else None,
                        elicitation_callback=self.elicitation_callback,
                        message_handler=self.message_handler,
                    )
                )
                server_result = await session.initialize()
                self.server_initialize_result = server_result
                self.session = session
                connection_succeeded = True
            except Exception as exc:
                http_error = self._extract_http_error_from_exception(exc)
                if http_error:
                    self._raise_user_error_for_http_error(http_error)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                if isinstance(exc, httpx.HTTPStatusError | httpx.ConnectError | httpx.TimeoutException):
                    self._raise_user_error_for_http_error(exc)
                raise
            finally:
                if not connection_succeeded:
                    try:
                        await self.cleanup()
                    except UserError:
                        raise
                    except Exception as cleanup_error:
                        if isinstance(cleanup_error, RuntimeError) and "cancel scope" in str(cleanup_error):
                            logger.debug(
                                "Ignoring cancel scope error during cleanup of MCP server "
                                f"'{self.name}': {cleanup_error}"
                            )
                        else:
                            logger.warning(
                                f"Error during cleanup of MCP server '{self.name}': {cleanup_error}"
                            )

        @asynccontextmanager
        async def _isolated_client_session(self) -> Any:
            from agents.mcp.server import ClientSession

            async with AsyncExitStack() as exit_stack:
                transport = await exit_stack.enter_async_context(self.create_streams())
                read, write, *_ = transport
                session = await exit_stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        timedelta(seconds=self.client_session_timeout_seconds)
                        if self.client_session_timeout_seconds
                        else None,
                        elicitation_callback=self.elicitation_callback,
                        message_handler=self.message_handler,
                    )
                )
                await session.initialize()
                yield session

    return ElicitationMCPServerStreamableHttp
