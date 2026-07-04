from __future__ import annotations

from dataclasses import dataclass

from diffbot_agent.session_usage import ModelPricing, SessionUsage


@dataclass
class _Details:
    cached_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class _Usage:
    requests: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    input_tokens_details: _Details | None = None
    output_tokens_details: _Details | None = None


PRICES = {"gpt-5.1": ModelPricing(input_per_mtok=2.0, output_per_mtok=8.0)}


def test_add_accumulates_tokens_and_requests() -> None:
    usage = SessionUsage("gpt-5.1", PRICES)
    usage.add(_Usage(input_tokens=1000, output_tokens=500))
    usage.add(_Usage(input_tokens=2000, output_tokens=1500))

    assert usage.requests == 2
    assert usage.input_tokens == 3000
    assert usage.output_tokens == 2000
    assert usage.total_tokens == 5000


def test_cost_matches_price_table() -> None:
    usage = SessionUsage("gpt-5.1", PRICES)
    usage.add(_Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    # 1M input @ $2 + 1M output @ $8
    assert usage.cost_usd == 10.0


def test_unknown_or_local_model_is_free() -> None:
    usage = SessionUsage("llama-local", PRICES)
    usage.add(_Usage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert usage.total_tokens == 2_000_000
    assert usage.cost_usd == 0.0


def test_none_and_empty_usage_are_ignored() -> None:
    usage = SessionUsage("gpt-5.1", PRICES)
    usage.add(None)
    usage.add(_Usage(requests=0, input_tokens=0, output_tokens=0))
    assert usage.requests == 0
    assert usage.total_tokens == 0
    assert usage.cost_usd == 0.0


def test_delta_and_reset() -> None:
    usage = SessionUsage("gpt-5.1", PRICES)
    usage.add(_Usage(input_tokens=100, output_tokens=50))
    snapshot = usage.snapshot()
    usage.add(_Usage(input_tokens=300, output_tokens=100))

    delta = usage.delta_since(snapshot)
    assert delta.input_tokens == 300
    assert delta.output_tokens == 100

    usage.reset()
    assert usage.total_tokens == 0
    assert usage.cost_usd == 0.0
    assert usage.context_tokens == 0


def test_context_tracks_latest_request() -> None:
    usage = SessionUsage("gpt-5.1", PRICES, context_limit=1000)
    usage.add(_Usage(input_tokens=200, output_tokens=50))
    usage.add(_Usage(input_tokens=800, output_tokens=100))
    # Gauge reflects the most recent request, not the cumulative sum.
    assert usage.context_tokens == 800
    assert "800" in usage.context_text()
    assert "80%" in usage.context_text()


def test_context_without_limit_shows_bare_count() -> None:
    usage = SessionUsage("llama-local")
    usage.add(_Usage(input_tokens=1500, output_tokens=100))
    assert usage.context_tokens == 1500
    assert "/" not in usage.context_text()


def test_render_helpers_are_nonempty() -> None:
    usage = SessionUsage("gpt-5.1", PRICES)
    baseline = usage.snapshot()
    usage.add(
        _Usage(
            input_tokens=12345,
            output_tokens=6789,
            input_tokens_details=_Details(cached_tokens=1000),
            output_tokens_details=_Details(reasoning_tokens=500),
        )
    )

    assert "gpt-5.1" in usage.toolbar_text()
    assert usage.turn_summary(usage.delta_since(baseline))
    assert "cached 1,000" in usage.breakdown()
    assert "reasoning 500" in usage.breakdown()
