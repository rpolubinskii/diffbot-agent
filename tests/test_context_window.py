from __future__ import annotations

from diffbot_agent.context_window import (
    IMAGE_TOKEN_ESTIMATE,
    _estimate_tokens,
    trim_to_token_budget,
)


def test_preamble_retained_and_oldest_trimmed() -> None:
    items = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "A" * 4000},  # ~1000 tokens (oldest)
        {"role": "user", "content": "B" * 4000},  # ~1000 tokens
        {"role": "user", "content": "C"},  # newest, tiny
    ]

    out = trim_to_token_budget(items, 1100)
    contents = [i.get("content", "") for i in out]

    assert out[0] == {"role": "system", "content": "sys"}  # preamble kept verbatim
    assert "C" in contents  # newest kept
    assert not any(c.startswith("A") for c in contents)  # oldest dropped to fit budget


def test_zero_or_negative_budget_keeps_everything() -> None:
    items = [{"role": "user", "content": f"m{n}"} for n in range(5)]
    assert len(trim_to_token_budget(items, 0)) == 5
    assert len(trim_to_token_budget(items, -1)) == 5


def test_keeps_at_least_one_body_item_over_budget() -> None:
    items = [{"role": "user", "content": "X" * 40000}]  # single item far over budget
    assert len(trim_to_token_budget(items, 100)) == 1


def test_images_charged_flat_cost() -> None:
    image_item = {
        "type": "function_call_output",
        "call_id": "c",
        "output": {"image_url": "data:image/png;base64," + "A" * 100000},
    }
    text_item = {"role": "user", "content": "hello world"}

    assert _estimate_tokens(image_item) == IMAGE_TOKEN_ESTIMATE  # not chars/4 of the blob
    assert 0 < _estimate_tokens(text_item) < 100


def test_orphaned_tool_pairing_is_repaired() -> None:
    body = [
        {"type": "function_call", "call_id": "a", "name": "t", "arguments": "x" * 8000},
        {"type": "function_call_output", "call_id": "a", "output": "ra"},
        {"type": "function_call", "call_id": "b", "name": "t", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "b", "output": "rb"},
    ]

    out = trim_to_token_budget(body, 500)
    kept = [(i.get("type"), i.get("call_id")) for i in out]

    # The huge call_a is dropped, and its now-orphaned output is repaired away.
    assert ("function_call", "a") not in kept
    assert ("function_call_output", "a") not in kept
    # The b call/output pair survives intact.
    assert ("function_call", "b") in kept
    assert ("function_call_output", "b") in kept
