from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone

from diffbot_agent.episode import build_canonical_record
from diffbot_agent.memory_backend import (
    EMPTY_MEMORY,
    DiffbotMcpMemoryBackend,
    render_facts,
)


class FakeClient:
    def __init__(self, *, recall_result=None, fail_start=False, raise_on=()):
        self.started = False
        self.stopped = False
        self.calls: list[tuple[str, dict]] = []
        self._recall_result = recall_result
        self._fail_start = fail_start
        self._raise_on = set(raise_on)

    async def start(self) -> None:
        if self._fail_start:
            raise RuntimeError("no connection")
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if name in self._raise_on:
            raise RuntimeError("call failed")
        if name == "memory.recall":
            return self._recall_result
        return {"ok": True}


_NOW = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)


def _record():
    return build_canonical_record(
        session_id="s",
        started_at="2026-06-23T11:59:00+00:00",
        completed_at="2026-06-23T11:59:30+00:00",
        command="drive to the kitchen",
        completion_status="completed",
        items=[
            {"type": "function_call", "call_id": "n1", "name": "x__nav.move_to", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "n1", "output": '{"success":true}'},
        ],
        final_output="done",
    )


class RenderFactsTest(unittest.TestCase):
    def test_structured_facts_with_validity(self) -> None:
        result = {
            "structuredContent": {
                "facts": [
                    {"fact": "the dock is by the window", "valid_at": "2026-06-20T00:00:00Z"},
                    {"fact": "the user prefers metric", "valid_at": None, "invalid_at": None},
                ]
            }
        }
        rendered = render_facts(result)
        self.assertIn("[2026-06-20T00:00:00Z] the dock is by the window", rendered)
        self.assertIn("the user prefers metric", rendered)

    def test_text_json_content(self) -> None:
        result = {"content": [{"type": "text", "text": '{"facts": [{"fact": "kitchen has a fridge"}]}'}]}
        self.assertEqual(render_facts(result), "kitchen has a fridge")

    def test_error_result_is_none(self) -> None:
        self.assertEqual(render_facts({"structuredContent": {"ok": False, "error_class": "memory_unavailable"}}), EMPTY_MEMORY)

    def test_empty_is_none(self) -> None:
        self.assertEqual(render_facts({"structuredContent": {"facts": []}}), EMPTY_MEMORY)


class DiffbotMcpMemoryBackendTest(unittest.TestCase):
    def test_recall_renders_facts(self) -> None:
        fake = FakeClient(recall_result={"structuredContent": {"facts": [{"fact": "dock is by the window"}]}})
        backend = DiffbotMcpMemoryBackend("url", client=fake)

        async def run() -> str:
            await backend.start()
            out = await backend.recall(query="where is the dock", limit=10, now=_NOW)
            await backend.close()
            return out

        self.assertEqual(asyncio.run(run()), "dock is by the window")

    def test_add_episode_writes_remember(self) -> None:
        fake = FakeClient()
        backend = DiffbotMcpMemoryBackend("url", client=fake)

        async def run() -> None:
            await backend.start()
            await backend.add_episode(_record())
            await backend.close()  # gathers the fire-and-forget write

        asyncio.run(run())
        remembered = [args for name, args in fake.calls if name == "memory.remember"]
        self.assertEqual(len(remembered), 1)
        self.assertIn("drive to the kitchen", remembered[0]["content"])

    def test_recall_degrades_to_none_on_error(self) -> None:
        fake = FakeClient(raise_on={"memory.recall"})
        backend = DiffbotMcpMemoryBackend("url", client=fake)

        async def run() -> str:
            await backend.start()
            out = await backend.recall(query="x", limit=10, now=_NOW)
            await backend.close()
            return out

        self.assertEqual(asyncio.run(run()), EMPTY_MEMORY)

    def test_not_connected_is_inert(self) -> None:
        fake = FakeClient(fail_start=True)
        backend = DiffbotMcpMemoryBackend("url", client=fake)

        async def run() -> str:
            await backend.start()  # connection fails, swallowed
            await backend.add_episode(_record())  # no-op
            out = await backend.recall(query="x", limit=10, now=_NOW)
            await backend.close()
            return out

        self.assertEqual(asyncio.run(run()), EMPTY_MEMORY)
        self.assertEqual(fake.calls, [])  # nothing attempted while disconnected

    def test_reset_is_non_destructive(self) -> None:
        fake = FakeClient()
        backend = DiffbotMcpMemoryBackend("url", client=fake)

        async def run() -> None:
            await backend.start()
            await backend.reset()
            await backend.close()

        asyncio.run(run())
        self.assertEqual(fake.calls, [])  # reset never clears the graph


if __name__ == "__main__":
    unittest.main()
