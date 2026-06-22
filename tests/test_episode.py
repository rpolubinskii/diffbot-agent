from __future__ import annotations

import json
import unittest

from diffbot_agent.episode import (
    _normalize_tool_name,
    _tool_category,
    build_canonical_record,
    set_mcp_tool_categories,
    set_tool_category_overrides,
)


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def _output(call_id: str, output: object) -> dict[str, object]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


class ToolClassificationTest(unittest.TestCase):
    """Resolution order: config override > mcp-provided > substring heuristic."""

    def tearDown(self) -> None:
        set_mcp_tool_categories({})
        set_tool_category_overrides({})

    def test_mcp_provided_categories_are_used(self) -> None:
        set_mcp_tool_categories(
            {
                "diffbot-mcp__nav.move_to": "navigation",
                "diffbot-mcp__speak.say": "speech",
                "diffbot-mcp__vision.get_camera_image": "vision",
                "diffbot-mcp__nav.get_pose": "status",
            }
        )
        for raw, expected in {
            "diffbot-mcp__nav.move_to": "navigation",
            "diffbot-mcp__speak.say": "speech",
            "diffbot-mcp__vision.get_camera_image": "vision",
            "diffbot-mcp__nav.get_pose": "status",
        }.items():
            self.assertEqual(_tool_category(_normalize_tool_name(raw)), expected, raw)

    def test_mcp_category_wins_over_heuristic(self) -> None:
        # Heuristic would say "navigation" (contains "move"); mcp says "status".
        set_mcp_tool_categories({"x__move_probe": "status"})
        self.assertEqual(_tool_category(_normalize_tool_name("x__move_probe")), "status")

    def test_override_wins_over_mcp(self) -> None:
        set_mcp_tool_categories({"x__foo": "navigation"})
        set_tool_category_overrides({"foo": "safety"})
        self.assertEqual(_tool_category(_normalize_tool_name("x__foo")), "safety")

    def test_unmapped_tool_defaults_to_tool(self) -> None:
        self.assertEqual(_tool_category(_normalize_tool_name("diffbot-mcp__totally_new")), "tool")

    def test_heuristic_fallback_when_no_meta(self) -> None:
        # A movement tool that shipped without category meta still invalidates the
        # camera image via the heuristic safety net.
        self.assertEqual(
            _tool_category(_normalize_tool_name("diffbot-mcp__drive_somewhere")), "navigation"
        )


class CanonicalRecordTest(unittest.TestCase):
    def test_status_tool_dropped_navigation_kept_error_captured(self) -> None:
        items = [
            _call("p1", "x__get_pose", {}),
            _output("p1", '{"pose":{"x":1},"imu":{}}'),
            _call("n1", "x__navigate_to", {"x": 2}),
            _output("n1", '{"success":true,"state":"succeeded"}'),
            _call("d1", "x__dock", {}),
            _output("d1", '{"success":false,"error":"dock not found"}'),
        ]
        record = build_canonical_record(
            session_id="s",
            started_at="2026-06-22T12:00:00+00:00",
            completed_at="2026-06-22T12:00:05+00:00",
            command="go dock",
            completion_status="completed",
            items=items,
            final_output="tried",
        )
        ops = {event["operation"] for event in record.tool_events}
        self.assertNotIn("get_pose", ops)  # status tools are pruned
        self.assertIn("navigate_to", ops)
        self.assertTrue(record.navigation_outcomes)
        self.assertTrue(record.error_outcomes)
        self.assertTrue(any("dock not found" in str(e) for e in record.error_outcomes))

    def test_runtime_error_recorded(self) -> None:
        record = build_canonical_record(
            session_id="s",
            started_at="2026-06-22T12:00:00+00:00",
            completed_at="2026-06-22T12:00:05+00:00",
            command="x",
            completion_status="failed",
            items=[],
            error=RuntimeError("boom"),
        )
        self.assertTrue(any(e.get("error") == "boom" for e in record.error_outcomes))


if __name__ == "__main__":
    unittest.main()
