from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from a2a.helpers.proto_helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from diffbot_agent.agent_runtime import AgentRuntime, TurnResult
from diffbot_agent.runtime_controller import RuntimeController


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0


def create_app(
    runtime: AgentRuntime,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Starlette:
    controller = RuntimeController(runtime)
    public_url = f"http://{host}:{port}"
    agent_card = _agent_card(public_url)
    request_handler = DefaultRequestHandler(
        agent_executor=DiffbotAgentExecutor(controller),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    mcp_app = _mcp_app(controller)

    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", **controller.status()})

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        await controller.start()
        try:
            async with mcp_app.router.lifespan_context(mcp_app):
                yield
        finally:
            await controller.stop()

    routes = [Route("/health", health, methods=["GET"])]
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, "/"))
    routes.extend(mcp_app.routes)
    return Starlette(routes=routes, lifespan=lifespan)


class DiffbotAgentExecutor(AgentExecutor):
    def __init__(self, controller: RuntimeController) -> None:
        self._controller = controller

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.current_task is not None:
            task = context.current_task
        elif context.message is not None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        else:
            return

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task.id,
            context_id=task.context_id,
        )
        command = context.get_user_input().strip()
        if not command:
            await updater.reject(
                new_text_message(
                    "No text command was provided.",
                    context_id=task.context_id,
                    task_id=task.id,
                )
            )
            return

        await updater.start_work(
            new_text_message(
                "DiffBot command accepted.",
                context_id=task.context_id,
                task_id=task.id,
            )
        )
        result = await self._controller.run_command(
            command,
            timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        )
        await _publish_a2a_result(updater, result)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not context.task_id or not context.context_id:
            return
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel(
            new_text_message(
                "Cancellation requested.",
                context_id=context.context_id,
                task_id=context.task_id,
            )
        )


async def _publish_a2a_result(updater: TaskUpdater, result: TurnResult) -> None:
    if result.final_text:
        await updater.add_artifact(
            parts=[new_text_part(result.final_text, media_type="text/plain")],
            name="diffbot-agent-result",
            last_chunk=True,
        )

    message_text = result.final_text or result.error or result.status
    message = new_text_message(
        message_text,
        context_id=updater.context_id,
        task_id=updater.task_id,
    )
    if result.status == "completed":
        await updater.complete(message)
    elif result.status == "busy":
        await updater.reject(message)
    else:
        await updater.failed(message)


def _mcp_app(controller: RuntimeController) -> Starlette:
    mcp = FastMCP(
        "diffbot-agent",
        instructions=(
            "Use diffbot_agent.run_command to delegate robot-facing requests to "
            "the long-running DiffBot agent. The command result is returned to "
            "the MCP caller as text/structured data."
        ),
        streamable_http_path="/mcp",
    )

    @mcp.tool(
        name="diffbot_agent.run_command",
        description="Run one command through the long-running DiffBot agent.",
    )
    async def run_command(
        command: str,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        result = await controller.run_command(
            command,
            timeout_seconds=timeout_seconds,
        )
        return result.compact_dict()

    @mcp.tool(
        name="diffbot_agent.status",
        description="Return the local DiffBot agent server status and session usage.",
    )
    def status() -> dict[str, Any]:
        return controller.status()

    @mcp.tool(
        name="diffbot_agent.reset_session",
        description="Reset the DiffBot agent conversation session.",
    )
    async def reset_session() -> dict[str, Any]:
        return (await controller.reset()).compact_dict()

    return mcp.streamable_http_app()


def _agent_card(public_url: str) -> AgentCard:
    skill = AgentSkill(
        id="diffbot_agent_command",
        name="DiffBot Agent Command",
        description=(
            "Delegate text commands to the long-running DiffBot robot agent and "
            "receive the final textual result."
        ),
        input_modes=["text/plain"],
        output_modes=["text/plain"],
        tags=["diffbot", "robot", "mcp", "a2a"],
        examples=[
            "Summarize the current robot status.",
            "Describe what the robot can see.",
            "Check diagnostics and report any actionable failures.",
        ],
    )
    return AgentCard(
        name="DiffBot Agent",
        description="Long-running DiffBot robot control agent.",
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                url=public_url,
            )
        ],
        skills=[skill],
    )
