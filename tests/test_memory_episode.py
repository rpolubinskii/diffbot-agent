from __future__ import annotations

import asyncio
import json
from typing import Any

from diffbot_agent.episode import (
    UNKNOWN_OUTPUT_PREVIEW_CHARS,
    CanonicalCommandRecord,
    build_canonical_record,
    build_memory_episode_draft,
    set_mcp_tool_categories,
)
from diffbot_agent.memory_backend import DiffbotMcpMemoryBackend


def setup_function() -> None:
    set_mcp_tool_categories({})


def _record(
    *,
    command: str,
    tool_events: tuple[dict[str, Any], ...] = (),
    completion_status: str = "completed",
    final_assistant_text: str = "",
) -> CanonicalCommandRecord:
    return CanonicalCommandRecord(
        session_id="test-session",
        started_at="2026-07-04T10:00:00+00:00",
        completed_at="2026-07-04T10:00:01+00:00",
        command=command,
        completion_status=completion_status,
        final_assistant_text=final_assistant_text,
        tool_events=tool_events,
    )


def test_tool_noise_is_removed_but_command_still_goes_to_extractor() -> None:
    record = _record(
        command="check your status and wait five seconds",
        tool_events=(
            {"sequence": 1, "tool": "robot.get_diagnostics", "category": "status"},
            {"sequence": 2, "tool": "system.wait", "category": "tool"},
            {"sequence": 3, "tool": "speak.say", "category": "speech"},
            {"sequence": 4, "tool": "memory.recall", "category": "tool"},
        ),
        final_assistant_text="Done.",
    )

    draft = build_memory_episode_draft(record)

    assert draft is not None
    assert draft["command"] == "check your status and wait five seconds"
    assert draft["final_assistant_text"] == "Done."
    assert "tool_events" not in draft


def test_navigation_success_or_failure_is_sent_for_extractor_judgment() -> None:
    navigation_event = (
        {
            "sequence": 1,
            "tool": "nav.move_to",
            "category": "navigation",
            "arguments": {"x": 1.0, "y": 2.0, "yawRadians": 0.0},
            "output": '{"status":"succeeded"}',
        },
    )

    success_draft = build_memory_episode_draft(
        _record(command="go to the kitchen", tool_events=navigation_event)
    )
    failure_draft = build_memory_episode_draft(
        _record(
            command="go to the kitchen",
            tool_events=navigation_event,
            completion_status="failed",
        )
    )

    assert success_draft is not None
    assert success_draft["tool_events"] == [
        {
            "tool": "nav.move_to",
            "category": "navigation",
            "arguments": {"x": 1.0, "y": 2.0, "yawRadians": 0.0},
            "output": '{"status":"succeeded"}',
        }
    ]
    assert failure_draft is not None
    assert failure_draft["completion_status"] == "failed"


def test_user_preference_turn_is_not_preclassified_by_agent() -> None:
    draft = build_memory_episode_draft(
        _record(
            command="Remember that I prefer short spoken status updates.",
            final_assistant_text="I will remember that.",
        )
    )

    assert draft is not None
    assert draft["command"] == "Remember that I prefer short spoken status updates."
    assert draft["completion_status"] == "completed"
    assert "learned_fact_signals" not in draft
    assert "tool_events" not in draft


def test_spatial_fact_and_semantic_observation_are_not_preclassified_by_agent() -> None:
    draft = build_memory_episode_draft(
        _record(
            command="The charging dock is by the window.",
            tool_events=(
                {
                    "sequence": 1,
                    "tool": "semantic.find",
                    "category": "semantic",
                    "arguments": {"query": "charging dock"},
                    "output": '{"matches":[{"label":"charging dock","x":1.2,"y":-0.4}]}',
                },
            ),
        )
    )

    assert draft is not None
    assert "learned_fact_signals" not in draft
    assert draft["tool_events"][0]["tool"] == "semantic.find"
    assert "charging dock" in draft["tool_events"][0]["output"]


def test_image_tool_payloads_are_sanitized_and_bounded() -> None:
    set_mcp_tool_categories({"vision.get_camera_image": "vision"})
    image = "data:image/png;base64," + ("A" * 600)
    output_text = "mug on the table " + ("x" * 900)
    record = build_canonical_record(
        session_id="test-session",
        started_at="2026-07-04T10:00:00+00:00",
        completed_at="2026-07-04T10:00:01+00:00",
        command="look for objects on the table",
        completion_status="completed",
        items=[
            {
                "type": "function_call",
                "name": "vision.get_camera_image",
                "call_id": "call_1",
                "arguments": json.dumps({"image_url": image, "note": "inspect table"}),
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"description": output_text, "image_url": image},
            },
        ],
    )

    draft = build_memory_episode_draft(record)

    assert draft is not None
    encoded = json.dumps(draft)
    assert "data:image" not in encoded
    assert "call_1" not in encoded
    tool_event = draft["tool_events"][0]
    assert tool_event["arguments"] == {"note": "inspect table"}
    assert len(tool_event["output"]) <= UNKNOWN_OUTPUT_PREVIEW_CHARS + 3


def test_memory_backend_skips_remember_when_draft_is_empty() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any] | None]] = []

        async def start(self) -> None:
            return None

        async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> None:
            self.calls.append((tool_name, arguments))

        async def stop(self) -> None:
            return None

    async def run() -> None:
        client = FakeClient()
        backend = DiffbotMcpMemoryBackend("http://unused.example/mcp", client=client)
        await backend.start()
        await backend.add_episode(_record(command=""))
        await backend.close()
        assert client.calls == []

    asyncio.run(run())


def test_memory_backend_sends_structured_draft_as_json_type() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any] | None]] = []

        async def start(self) -> None:
            return None

        async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> None:
            self.calls.append((tool_name, arguments))

        async def stop(self) -> None:
            return None

    async def run() -> None:
        client = FakeClient()
        backend = DiffbotMcpMemoryBackend("http://unused.example/mcp", client=client)
        await backend.start()
        await backend.add_episode(_record(command="Remember that I prefer short updates."))
        await backend.close()

        assert len(client.calls) == 1
        tool_name, arguments = client.calls[0]
        assert tool_name == "memory.remember"
        assert arguments is not None
        assert arguments["type"] == "json"
        assert json.loads(arguments["content"])["command"] == "Remember that I prefer short updates."

    asyncio.run(run())
