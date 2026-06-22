from __future__ import annotations

import copy
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from diffbot_agent.context_window import CommandContextState, compose_command_input
from diffbot_agent.episode import build_canonical_record, render_recent_memories
from diffbot_agent.sanitize import IMAGE_PLACEHOLDER, contains_image


class MutableClock:
    def __init__(self, monotonic: float, utc: datetime):
        self.monotonic = monotonic
        self.utc = utc

    def monotonic_now(self) -> float:
        return self.monotonic

    def utc_now(self) -> datetime:
        return self.utc


class CommandMemoryTest(unittest.TestCase):
    def test_active_command_and_tool_timestamp_are_stable(self) -> None:
        clock = MutableClock(
            112.4,
            datetime(2026, 6, 11, 18, 6, 22, tzinfo=timezone.utc),
        )
        command = "Tell me where the refrigerator is"
        state = CommandContextState(
            full_tool_rounds=6,
            command=command,
            robot_status="idle",
            started_at="2026-06-11T18:06:10+00:00",
            started_monotonic=100.0,
            monotonic_clock=clock.monotonic_now,
            utc_clock=clock.utc_now,
        )
        items = [
            _user_item("original input"),
            _call("speech-1", "diffbot-mcp__speak", {"text": "Working on it"}),
            _output("speech-1", '{"ok":true}'),
        ]
        original = copy.deepcopy(items)

        first = state.filter_model_input(_payload(items)).input
        clock.monotonic = 500.0
        clock.utc = datetime(2026, 6, 11, 19, 0, tzinfo=timezone.utc)
        second = state.filter_model_input(_payload(items)).input

        self.assertIn(f"[+0.0s | 18:06:10Z] {command}", first[0]["content"])
        self.assertIn(f"[+0.0s | 18:06:10Z] {command}", second[0]["content"])
        self.assertEqual(first[2]["output"], second[2]["output"])
        self.assertTrue(first[2]["output"].startswith("[+12.4s | 18:06:22Z]"))
        self.assertEqual(items, original)

    def test_image_survives_speech_then_navigation_invalidates_it(self) -> None:
        clock = MutableClock(
            101.0,
            datetime(2026, 6, 11, 18, 6, 11, tzinfo=timezone.utc),
        )
        state = CommandContextState(
            full_tool_rounds=6,
            command="Find the refrigerator",
            robot_status="idle",
            started_at="2026-06-11T18:06:10+00:00",
            started_monotonic=100.0,
            monotonic_clock=clock.monotonic_now,
            utc_clock=clock.utc_now,
        )
        first_image = _image_content("AAAA")
        items = [
            _user_item("input"),
            _call("camera-1", "diffbot-mcp__capture_camera", {}),
            _output("camera-1", first_image),
            _call("speech-1", "diffbot-mcp__speak", {"text": "I can see it"}),
            _output("speech-1", '{"ok":true}'),
        ]

        after_speech = state.filter_model_input(_payload(items)).input
        self.assertTrue(contains_image(_find_output(after_speech, "camera-1")))

        items.extend(
            [
                _call("nav-1", "diffbot-mcp__navigate_to", {"x": 1}),
                _output("nav-1", '{"success":true}'),
            ]
        )
        after_navigation = state.filter_model_input(_payload(items)).input
        self.assertFalse(contains_image(_find_output(after_navigation, "camera-1")))
        self.assertIn(IMAGE_PLACEHOLDER, str(_find_output(after_navigation, "camera-1")))

        clock.monotonic = 109.0
        clock.utc = datetime(2026, 6, 11, 18, 6, 19, tzinfo=timezone.utc)
        second_image = _image_content("BBBB")
        items.extend(
            [
                _call("camera-2", "diffbot-mcp__capture_camera", {}),
                _output("camera-2", second_image),
            ]
        )
        after_new_image = state.filter_model_input(_payload(items)).input
        self.assertFalse(contains_image(_find_output(after_new_image, "camera-1")))
        self.assertTrue(contains_image(_find_output(after_new_image, "camera-2")))
        self.assertIn("[+9.0s | 18:06:19Z]", str(_find_output(after_new_image, "camera-2")))

    def test_latest_image_survives_tool_round_compaction(self) -> None:
        state = CommandContextState(
            full_tool_rounds=1,
            command="Inspect the room",
            robot_status="idle",
            started_at="2026-06-11T18:06:10+00:00",
            started_monotonic=0.0,
            monotonic_clock=lambda: 3.0,
            utc_clock=lambda: datetime(2026, 6, 11, 18, 6, 13, tzinfo=timezone.utc),
        )
        items = [
            _user_item("input"),
            _call("camera-1", "diffbot-mcp__capture_camera", {}),
            _output("camera-1", _image_content("AAAA")),
            _call("speech-1", "diffbot-mcp__speak", {"text": "First update"}),
            _output("speech-1", '{"ok":true}'),
            _call("speech-2", "diffbot-mcp__speak", {"text": "Second update"}),
            _output("speech-2", '{"ok":true}'),
        ]

        filtered = state.filter_model_input(_payload(items)).input

        self.assertTrue(contains_image(_find_output(filtered, "camera-1")))

    def test_history_order_age_stale_hint_and_deduplication(self) -> None:
        claim = "The refrigerator is in the kitchen."
        items = [
            _user_item("input"),
            _call("camera-1", "diffbot-mcp__capture_camera", {}),
            _output("camera-1", _image_content("AAAA")),
            _call("speech-1", "diffbot-mcp__speak", {"text": claim}),
            _output("speech-1", '{"ok":true}'),
            {
                "role": "assistant",
                "type": "message",
                "content": [{"type": "output_text", "text": claim}],
            },
        ]
        record = build_canonical_record(
            session_id="test",
            started_at="2026-06-11T18:05:20+00:00",
            completed_at="2026-06-11T18:05:25+00:00",
            command="Where is the refrigerator?",
            completion_status="completed",
            items=items,
            final_output=claim,
        )

        self.assertEqual(record.spoken_text, ())
        self.assertFalse(any(event.get("category") == "speech" for event in record.tool_events))
        self.assertFalse(any("capture_camera" in str(event) for event in record.tool_events))
        self.assertFalse(any(IMAGE_PLACEHOLDER in str(event) for event in record.tool_events))

        rendered = render_recent_memories(
            [record],
            now=datetime(2026, 6, 11, 18, 6, 9, tzinfo=timezone.utc),
        )
        self.assertEqual(rendered.count(claim), 1)
        self.assertIn("[18:05:25Z | age 44s]", rendered)
        self.assertIn("STALE_HINTS_REQUIRING_REVERIFICATION", rendered)
        self.assertIn("cannot establish current visibility", rendered)

        prompt = compose_command_input(
            "Look again",
            "stopped",
            rendered,
            started_at="2026-06-11T18:06:09+00:00",
        )
        headings = [
            prompt.index("CURRENT_COMMAND"),
            prompt.index("CURRENT_ROBOT_STATUS"),
            prompt.index("CURRENT_COMMAND_TIMELINE"),
            prompt.index("HISTORICAL_MEMORY"),
        ]
        self.assertEqual(headings, sorted(headings))

    def test_new_command_state_does_not_inherit_an_image(self) -> None:
        previous = CommandContextState(
            full_tool_rounds=6,
            command="First command",
            robot_status="idle",
            started_at="2026-06-11T18:06:10+00:00",
            started_monotonic=0.0,
            monotonic_clock=lambda: 1.0,
            utc_clock=lambda: datetime(2026, 6, 11, 18, 6, 11, tzinfo=timezone.utc),
        )
        previous.filter_model_input(
            _payload(
                [
                    _user_item("input"),
                    _call("camera-1", "diffbot-mcp__capture_camera", {}),
                    _output("camera-1", _image_content("AAAA")),
                ]
            )
        )

        current = CommandContextState(
            full_tool_rounds=6,
            command="Second command",
            robot_status="idle",
            started_at="2026-06-11T18:07:10+00:00",
        )
        filtered = current.filter_model_input(_payload([_user_item("new input")])).input
        self.assertFalse(any(contains_image(item) for item in filtered))
        self.assertIn("Second command", filtered[0]["content"])
        self.assertNotIn("First command", str(filtered))


def _payload(items: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        model_data=SimpleNamespace(input=items, instructions="instructions")
    )


def _user_item(text: str) -> dict[str, object]:
    return {"role": "user", "type": "message", "content": text}


def _call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    import json

    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


def _output(call_id: str, output: object) -> dict[str, object]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def _image_content(data: str) -> list[dict[str, str]]:
    return [{"type": "input_image", "image_url": f"data:image/png;base64,{data}"}]


def _find_output(items: list[dict[str, object]], call_id: str) -> object:
    for item in items:
        if item.get("type") == "function_call_output" and item.get("call_id") == call_id:
            return item.get("output")
    raise AssertionError(f"No output for {call_id}")


if __name__ == "__main__":
    unittest.main()
