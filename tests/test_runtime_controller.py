from __future__ import annotations

import asyncio

from diffbot_agent.agent_runtime import TurnResult
from diffbot_agent.runtime_controller import RuntimeController
from diffbot_agent.session_usage import SessionUsage


class FakeRuntime:
    def __init__(self) -> None:
        self.usage = SessionUsage("fake-model")
        self.started = False
        self.stopped = False
        self.reset_count = 0
        self.commands: list[str] = []
        self.block = asyncio.Event()

    async def start(self) -> None:
        self.started = True

    async def run_turn(self, command: str) -> TurnResult:
        self.commands.append(command)
        if command == "block":
            await self.block.wait()
        return TurnResult(status="completed", final_text=f"done: {command}")

    async def reset(self) -> None:
        self.reset_count += 1

    async def stop(self) -> None:
        self.stopped = True


def test_runtime_controller_runs_command_and_reports_status() -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        controller = RuntimeController(runtime)

        await controller.start()
        result = await controller.run_command("  status please  ")
        status = controller.status()
        await controller.stop()

        assert runtime.started is True
        assert runtime.stopped is True
        assert runtime.commands == ["status please"]
        assert result.compact_dict()["final_text"] == "done: status please"
        assert status["started"] is True
        assert status["busy"] is False

    asyncio.run(scenario())


def test_runtime_controller_returns_busy_for_overlapping_command() -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        controller = RuntimeController(runtime)
        await controller.start()

        first = asyncio.create_task(controller.run_command("block"))
        while not controller.status()["busy"]:
            await asyncio.sleep(0)

        second = await controller.run_command("other")
        runtime.block.set()
        first_result = await first

        assert first_result.status == "completed"
        assert second.status == "busy"
        assert runtime.commands == ["block"]

    asyncio.run(scenario())
