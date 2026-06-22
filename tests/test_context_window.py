from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from diffbot_agent.context_window import (
    CommandContextState,
    compact_current_command,
    truncate_at_server_compaction,
)


def _payload(items: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        model_data=SimpleNamespace(input=items, instructions="instructions")
    )


def _user(text: str) -> dict[str, object]:
    return {"role": "user", "type": "message", "content": text}


def _call(call_id: str, name: str) -> dict[str, object]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _output(call_id: str, output: object) -> dict[str, object]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _types(items: list[dict[str, object]]) -> list[str]:
    return [i.get("type") for i in items if isinstance(i, dict)]


class ServerCompactionTest(unittest.TestCase):
    def test_truncate_keeps_from_last_marker(self) -> None:
        items = [
            _user("u"),
            _call("a", "t"),
            _output("a", "1"),
            {"type": "compaction"},
            _call("b", "t"),
            _output("b", "2"),
        ]
        result = truncate_at_server_compaction(items)
        self.assertEqual(_types(result), ["compaction", "function_call", "function_call_output"])

    def test_truncate_no_marker_is_unchanged(self) -> None:
        items = [_user("u"), _call("a", "t"), _output("a", "1")]
        self.assertEqual(len(truncate_at_server_compaction(items)), 3)

    def test_truncate_does_not_mutate_input(self) -> None:
        items = [_user("u"), {"type": "compaction"}, _call("b", "t"), _output("b", "2")]
        snapshot = json.dumps(items)
        truncate_at_server_compaction(items)
        self.assertEqual(json.dumps(items), snapshot)

    def test_filter_without_local_compaction_honours_server_marker(self) -> None:
        items = [
            _user("u"),
            _call("a", "t"),
            _output("a", "1"),
            {"type": "compaction"},
            _call("b", "t"),
            _output("b", "2"),
        ]
        state = CommandContextState(
            full_tool_rounds=1,
            command="c",
            robot_status="idle",
            compact_locally=False,
            started_at="2026-06-22T12:00:00+00:00",
        )
        out = state.filter_model_input(_payload(items)).input
        self.assertNotIn("a", [i.get("call_id") for i in out if isinstance(i, dict)])
        self.assertIn("b", [i.get("call_id") for i in out if isinstance(i, dict)])
        self.assertIn("CURRENT_COMMAND", out[0]["content"])
        self.assertIn("HISTORICAL_MEMORY", out[-1]["content"])


class LocalCompactionTest(unittest.TestCase):
    def test_old_rounds_compacted_when_over_budget(self) -> None:
        items = [
            _user("input"),
            _call("s1", "x__speak"),
            _output("s1", '{"ok":true}'),
            _call("s2", "x__speak"),
            _output("s2", '{"ok":true}'),
            _call("s3", "x__speak"),
            _output("s3", '{"ok":true}'),
        ]
        compacted = compact_current_command(items, full_tool_rounds=1)
        # Three rounds, keep the newest exact -> earlier rounds fold into one summary.
        texts = [i for i in compacted if i.get("type") == "message"]
        self.assertTrue(any("Compacted earlier tool rounds" in str(i) for i in texts))
        self.assertIn("s3", [i.get("call_id") for i in compacted if isinstance(i, dict)])

    def test_no_compaction_under_budget(self) -> None:
        items = [_user("input"), _call("s1", "x__speak"), _output("s1", '{"ok":true}')]
        self.assertEqual(compact_current_command(items, full_tool_rounds=6), items)


if __name__ == "__main__":
    unittest.main()
