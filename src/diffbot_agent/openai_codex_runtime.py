from __future__ import annotations

import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass

from diffbot_agent.config import AppConfig, ConfigError


INSTRUCTIONS = """You are DiffBot's long-running robot control agent.

You receive one operator or voice command per turn. Each turn includes fresh robot://status.
Use diffbot-mcp tools for robot state, navigation, vision, speech, and memory. Prefer high-level
tools such as nav.stop, nav.get_pose, vision.get_camera_image, and speak.say over low-level
diagnostics. When motion is uncertain, unsafe, interrupted, or failing, stop or cancel motion.
Do not invent robot state that was not provided in the turn or retrieved with a tool.
"""


@dataclass
class OpenAIAgentsRuntime:
    config: AppConfig

    def __post_init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._agent = None
        self._session = None
        self._run_config = None

    async def start(self) -> None:
        profile = self.config.agent
        if profile.backend == "openai" and not profile.openai_api_key.strip():
            raise ConfigError(f"[agents.{profile.name}].openai_api_key is required for OpenAI.")
        if profile.backend == "ollama" and (not profile.model.strip() or not profile.base_url.strip()):
            raise ConfigError(f"[agents.{profile.name}] requires model and base_url for Ollama.")

        from agents import Agent, RunConfig, SQLiteSession
        from agents import OpenAIProvider
        from agents import set_default_openai_key
        from agents.mcp import MCPServerStreamableHttp

        if profile.backend == "openai":
            set_default_openai_key(profile.openai_api_key)
            self._run_config = None
        elif profile.backend == "ollama":
            self._run_config = RunConfig(
                model_provider=OpenAIProvider(
                    api_key=profile.api_key or "ollama",
                    base_url=profile.base_url,
                    use_responses=False,
                )
            )
        else:
            raise ConfigError(f'Unsupported agent backend "{profile.backend}".')

        stack = AsyncExitStack()
        try:
            mcp_server = await stack.enter_async_context(
                MCPServerStreamableHttp(
                    name="diffbot-mcp",
                    params={
                        "url": self.config.mcp.url,
                        "timeout": 10,
                        "sse_read_timeout": 300,
                    },
                    cache_tools_list=True,
                    max_retry_attempts=2,
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
            run_config=self._run_config,
            session=self._session,
        )
        streamed_text = False

        async for event in result.stream_events():
            if _print_output_delta(event):
                streamed_text = True

        final_output = getattr(result, "final_output", None)
        if final_output and not streamed_text:
            print(final_output)
        elif streamed_text:
            print()
        sys.stdout.flush()

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._agent = None
            self._session = None
            self._run_config = None


OpenAICodexRuntime = OpenAIAgentsRuntime


def _ensure_robot_status(user_text: str, robot_status: str) -> str:
    if "Robot status:" in user_text or "robot://status" in user_text:
        return user_text
    return f"{user_text}\n\nFresh robot://status:\n{robot_status}"


def _print_output_delta(event: object) -> bool:
    if getattr(event, "type", None) != "raw_response_event":
        return False

    data = getattr(event, "data", None)
    if getattr(data, "type", None) != "response.output_text.delta":
        return False

    delta = getattr(data, "delta", "")
    if not delta:
        return False

    print(delta, end="", flush=True)
    return True
