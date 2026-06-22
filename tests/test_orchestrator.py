from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from diffbot_agent.orchestrator import Orchestrator, _is_reset_command


class ResetCommandMatchTest(unittest.TestCase):
    def test_reset_phrases_match(self) -> None:
        for phrase in (
            "reset context",
            "Reset the context",
            "clear memory.",
            "CLEAR MEMORY!",
            "forget everything",
            "  clear   the context  ",
        ):
            self.assertTrue(_is_reset_command(phrase), phrase)

    def test_non_reset_phrases_do_not_match(self) -> None:
        for phrase in (
            "stop",
            "drive to the reset context room",
            "describe what you can see",
            "remember the kitchen is here",
        ):
            self.assertFalse(_is_reset_command(phrase), phrase)


class _FakeRuntime:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.run_calls = 0

    async def reset(self) -> None:
        self.reset_calls += 1

    async def run_turn(self, command: str, robot_status: str) -> None:
        self.run_calls += 1


class _FakeMcp:
    def __init__(self) -> None:
        self.status_reads = 0

    async def read_robot_status(self) -> str:
        self.status_reads += 1
        return "idle"


def _orchestrator(runtime: _FakeRuntime, mcp: _FakeMcp, busy_policy: str = "ignore") -> Orchestrator:
    config = SimpleNamespace(agent_runtime=SimpleNamespace(busy_policy=busy_policy))
    return Orchestrator(config=config, runtime=runtime, mcp_client=mcp, audio_client=None)


class ResetRoutingTest(unittest.TestCase):
    def test_reset_command_routes_to_runtime_reset(self) -> None:
        runtime, mcp = _FakeRuntime(), _FakeMcp()
        orc = _orchestrator(runtime, mcp)
        asyncio.run(orc._accept_command("reset context"))
        self.assertEqual(runtime.reset_calls, 1)
        self.assertEqual(runtime.run_calls, 0)
        self.assertEqual(mcp.status_reads, 0)  # reset does not read robot status

    def test_normal_command_routes_to_turn(self) -> None:
        runtime, mcp = _FakeRuntime(), _FakeMcp()
        orc = _orchestrator(runtime, mcp)
        asyncio.run(orc._accept_command("drive forward"))
        self.assertEqual(runtime.run_calls, 1)
        self.assertEqual(runtime.reset_calls, 0)
        self.assertEqual(mcp.status_reads, 1)

    def test_command_ignored_while_busy(self) -> None:
        runtime, mcp = _FakeRuntime(), _FakeMcp()
        orc = _orchestrator(runtime, mcp)
        orc._turn_running = True
        asyncio.run(orc._accept_command("reset context"))
        self.assertEqual(runtime.reset_calls, 0)


if __name__ == "__main__":
    unittest.main()
