from __future__ import annotations

import asyncio
from types import SimpleNamespace

from diffbot_agent.command_memory import (
    IMAGE_PLACEHOLDER,
    CanonicalCommandRecord,
    CommandContextState,
    CommandMemoryStore,
    build_canonical_record,
    clear_command_memories,
    compact_current_command,
    compose_command_input,
    sanitize_session_items,
)
from diffbot_agent.logging_utils import redact


def _message(text: str, response_id: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
        "provider_data": {"response_id": response_id},
    }


def _round(index: int) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": f"reasoning {index}"}],
        },
        _message(f"assistant {index}", f"response-{index}"),
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__nav_turn",
            "call_id": call_id,
            "arguments": f'{{"radians":{index}}}',
        },
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": (
                f'{{"ok":true,"timestamp":"t{index}",'
                f'"goal_id":"goal-{index}","state":"succeeded"}}'
            ),
        },
    ]


def test_compact_current_command_keeps_latest_round_exact() -> None:
    initial = {"role": "user", "content": "command"}
    items = [initial, *_round(1), *_round(2), *_round(3)]

    compacted = compact_current_command(items, full_tool_rounds=1)

    assert compacted[0] == initial
    summary = compacted[1]["content"][0]["text"]
    assert "assistant 1" in summary
    assert "goal-2" in summary
    assert "reasoning 1" not in summary
    assert compacted[-4:] == _round(3)


def test_zero_full_rounds_keeps_newest_round_for_its_first_use() -> None:
    items = [{"role": "user", "content": "command"}, *_round(1), *_round(2)]

    compacted = compact_current_command(items, full_tool_rounds=0)

    assert len(compacted) == 6
    assert compacted[1]["type"] == "message"
    assert "Tool event:" in compacted[1]["content"][0]["text"]
    assert compacted[-4:] == _round(2)


def test_images_are_replayed_until_success_then_replaced() -> None:
    image_url = "data:image/jpeg;base64,QUJDRA=="
    items = [
        {"role": "user", "content": "look"},
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__vision_get_camera_image",
            "call_id": "camera-1",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "camera-1",
            "output": [
                {"type": "input_text", "text": "front camera"},
                {"type": "input_image", "image_url": image_url},
            ],
        },
    ]
    payload = SimpleNamespace(
        model_data=SimpleNamespace(input=items, instructions="instructions")
    )
    state = CommandContextState(full_tool_rounds=6)

    first = state.filter_model_input(payload)
    retry = state.filter_model_input(payload)
    assert image_url in str(first.input)
    assert image_url in str(retry.input)
    assert state.image_call_ids == {"camera-1"}

    state.mark_model_request_succeeded()
    consumed = state.filter_model_input(payload)
    assert image_url not in str(consumed.input)
    assert IMAGE_PLACEHOLDER in str(consumed.input)
    assert "front camera" in str(consumed.input)


def test_session_image_sanitizing_does_not_mutate_active_items() -> None:
    items = [
        {
            "type": "function_call_output",
            "call_id": "camera-1",
            "output": [
                {"type": "input_text", "text": "camera metadata"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,QUJD",
                },
            ],
        }
    ]

    sanitized = sanitize_session_items(items)

    assert "base64" in str(items)
    assert "base64" not in str(sanitized)
    assert "camera metadata" in str(sanitized)


def test_log_redaction_removes_image_payload_and_preserves_other_text() -> None:
    value = {
        "content": [
            {"type": "text", "text": "front camera"},
            {
                "type": "image",
                "data": "QUJDRA==",
                "mimeType": "image/jpeg",
            },
        ],
        "url": "data:image/png;base64,QUJD note",
    }

    redacted = redact(value)

    assert "QUJDRA==" not in str(redacted)
    assert "data:image" not in str(redacted)
    assert "front camera" in str(redacted)
    assert "note" in str(redacted)


def test_canonical_record_compacts_tool_events_and_excludes_status() -> None:
    long_output = "x" * 700
    items = [
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__speak_say",
            "call_id": "speech",
            "arguments": '{"text":"Turning now"}',
        },
        {
            "type": "function_call_output",
            "call_id": "speech",
            "output": '{"ok":true,"timestamp":"speech-time","ack":"large"}',
        },
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__nav_turn",
            "call_id": "nav",
            "arguments": '{"radians":1.57}',
        },
        {
            "type": "function_call_output",
            "call_id": "nav",
            "output": (
                '{"ok":true,"timestamp":"nav-time","goal_id":"goal-1",'
                '"ros":{"success":true,"state":"succeeded"}}'
            ),
        },
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__robot_status",
            "call_id": "status",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "status",
            "output": '{"pose":{"x":100,"y":200}}',
        },
        {
            "type": "function_call",
            "name": "mcp_diffbot-mcp__custom_probe",
            "call_id": "unknown",
            "arguments": '{"mode":"brief"}',
        },
        {
            "type": "function_call_output",
            "call_id": "unknown",
            "output": long_output,
        },
        _message("Done.", "final"),
    ]

    record = build_canonical_record(
        session_id="session",
        started_at="start",
        completed_at="complete",
        command="turn",
        completion_status="completed",
        items=items,
        final_output="Done.",
    )

    assert record.spoken_text == ("Turning now",)
    assert len(record.tool_events) == 3
    assert record.navigation_outcomes[0]["identifiers"]["goal_id"] == "goal-1"
    assert "pose" not in record.searchable_text
    unknown = next(event for event in record.tool_events if event["call_id"] == "unknown")
    assert unknown["output_preview"].endswith("...")
    assert len(unknown["output_preview"]) == 503
    assert "ack" not in str(record.tool_events[0])


def test_command_memory_store_returns_latest_in_chronological_order(tmp_path) -> None:
    async def exercise() -> None:
        db_path = tmp_path / "memory.sqlite3"
        store = CommandMemoryStore(db_path)
        try:
            for index in range(3):
                record = CanonicalCommandRecord(
                    session_id="session",
                    started_at=f"start-{index}",
                    completed_at=f"complete-{index}",
                    command=f"command-{index}",
                    completion_status="completed",
                    final_assistant_text="",
                    spoken_text=(),
                    tool_events=(),
                    navigation_outcomes=(),
                    safety_outcomes=(),
                    error_outcomes=(),
                    searchable_text=f"command-{index}",
                )
                await store.add(record)
            latest = await store.latest("session", 2)
            assert [record.command for record in latest] == ["command-1", "command-2"]
        finally:
            await store.close()

        clear_command_memories(db_path, "session")
        reopened = CommandMemoryStore(db_path)
        try:
            assert await reopened.latest("session", 4) == []
        finally:
            await reopened.close()

    asyncio.run(exercise())


def test_composed_input_marks_fresh_status_authoritative() -> None:
    text = compose_command_input("move forward", '{"pose":{"x":1}}', [])

    assert "Current operator command:\nmove forward" in text
    assert "Fresh robot://status (authoritative" in text
    assert "(none)" in text
