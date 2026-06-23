from __future__ import annotations

import asyncio
import unittest

from mcp import types

from diffbot_agent.elicitation import build_elicitation_callback
from diffbot_agent.operator_input import OperatorInputCoordinator, OperatorInputRoute


_RAW_ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class OperatorInputCoordinatorTest(unittest.TestCase):
    def test_pending_elicitation_consumes_next_input(self) -> None:
        async def run() -> None:
            coordinator = OperatorInputCoordinator()
            answer_task = asyncio.create_task(coordinator.request_answer(1.0))
            await asyncio.sleep(0)

            route = await coordinator.submit("  kitchen  ")

            self.assertIs(route, OperatorInputRoute.ELICITATION)
            self.assertEqual(await answer_task, "kitchen")
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(coordinator.next_command(), timeout=0.01)

        asyncio.run(run())

    def test_timeout_returns_none(self) -> None:
        async def run() -> None:
            coordinator = OperatorInputCoordinator()
            self.assertIsNone(await coordinator.request_answer(0.01))

        asyncio.run(run())

    def test_close_resolves_pending_answer_as_none(self) -> None:
        async def run() -> None:
            coordinator = OperatorInputCoordinator()
            answer_task = asyncio.create_task(coordinator.request_answer(1.0))
            await asyncio.sleep(0)

            await coordinator.close()

            self.assertIsNone(await answer_task)

        asyncio.run(run())


class ElicitationCallbackTest(unittest.TestCase):
    def test_accepts_single_answer_form(self) -> None:
        async def run() -> None:
            seen_timeout = None

            async def answer_provider(timeout_seconds: float) -> str | None:
                nonlocal seen_timeout
                seen_timeout = timeout_seconds
                return "yes"

            callback = build_elicitation_callback(answer_provider)
            result = await callback(
                None,
                types.ElicitRequestFormParams(
                    message="Continue?",
                    requestedSchema=_RAW_ANSWER_SCHEMA,
                    _meta={"timeoutSeconds": 3},
                ),
            )

            self.assertIsInstance(result, types.ElicitResult)
            self.assertEqual(result.action, "accept")
            self.assertEqual(result.content, {"answer": "yes"})
            self.assertEqual(seen_timeout, 3.0)

        asyncio.run(run())

    def test_none_answer_cancels(self) -> None:
        async def run() -> None:
            async def answer_provider(timeout_seconds: float) -> str | None:
                return None

            callback = build_elicitation_callback(answer_provider)
            result = await callback(
                None,
                types.ElicitRequestFormParams(
                    message="Continue?",
                    requestedSchema=_RAW_ANSWER_SCHEMA,
                ),
            )

            self.assertIsInstance(result, types.ElicitResult)
            self.assertEqual(result.action, "cancel")
            self.assertIsNone(result.content)

        asyncio.run(run())

    def test_rejects_unexpected_schema(self) -> None:
        async def run() -> None:
            async def answer_provider(timeout_seconds: float) -> str | None:
                return "unused"

            callback = build_elicitation_callback(answer_provider)
            result = await callback(
                None,
                types.ElicitRequestFormParams(
                    message="Continue?",
                    requestedSchema={
                        "type": "object",
                        "properties": {"comment": {"type": "string"}},
                        "required": ["comment"],
                    },
                ),
            )

            self.assertIsInstance(result, types.ErrorData)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
