from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModelPricing:
    """USD list price per 1M tokens for a model."""

    input_per_mtok: float
    output_per_mtok: float


# Placeholder list prices (USD per 1M tokens). Verify against current provider
# pricing and override per model in the config ``[pricing]`` table.
DEFAULT_MODEL_PRICES: dict[str, ModelPricing] = {
    "gpt-5.1": ModelPricing(input_per_mtok=1.25, output_per_mtok=10.0),
}


@dataclass(frozen=True)
class UsageSnapshot:
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class SessionUsage:
    """Accumulates token usage and estimated cost across a console session.

    Fed one SDK ``Usage`` per LLM call from the runtime's ``on_llm_end`` hook.
    Usage is duck-typed so this module stays free of the Agents SDK import.
    """

    def __init__(
        self,
        model: str,
        prices: Mapping[str, ModelPricing] | None = None,
        context_limit: int | None = None,
    ) -> None:
        self.model = model
        self._prices: dict[str, ModelPricing] = dict(prices or {})
        self.context_limit = context_limit
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        # Live gauge: input tokens of the most recent request ~= current context size.
        self.context_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, usage: Any) -> None:
        """Accumulate one ``Usage``; a ``None`` or all-zero usage is ignored."""
        if usage is None:
            return
        requests = _int(getattr(usage, "requests", 0))
        input_tokens = _int(getattr(usage, "input_tokens", 0))
        output_tokens = _int(getattr(usage, "output_tokens", 0))
        if not (requests or input_tokens or output_tokens):
            return
        self.requests += requests
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_input_tokens += _detail(usage, "input_tokens_details", "cached_tokens")
        self.reasoning_tokens += _detail(usage, "output_tokens_details", "reasoning_tokens")
        self.cost_usd += self._cost(input_tokens, output_tokens)
        if input_tokens:
            self.context_tokens = input_tokens

    def reset(self) -> None:
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.reasoning_tokens = 0
        self.cost_usd = 0.0
        self.context_tokens = 0

    def snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(self.requests, self.input_tokens, self.output_tokens, self.cost_usd)

    def delta_since(self, snapshot: UsageSnapshot) -> UsageSnapshot:
        return UsageSnapshot(
            requests=self.requests - snapshot.requests,
            input_tokens=self.input_tokens - snapshot.input_tokens,
            output_tokens=self.output_tokens - snapshot.output_tokens,
            cost_usd=self.cost_usd - snapshot.cost_usd,
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float:
        price = self._prices.get(self.model)
        if price is None:
            return 0.0
        return (
            input_tokens / 1_000_000 * price.input_per_mtok
            + output_tokens / 1_000_000 * price.output_per_mtok
        )

    def context_text(self) -> str:
        if self.context_limit:
            pct = self.context_tokens / self.context_limit * 100
            return f"{_fmt(self.context_tokens)}/{_fmt(self.context_limit)} ({pct:.0f}%)"
        return _fmt(self.context_tokens)

    def toolbar_text(self) -> str:
        return (
            f" {self.model} · ctx {self.context_text()}"
            f" · {_fmt(self.input_tokens)} in / {_fmt(self.output_tokens)} out"
            f" · {_fmt(self.total_tokens)} tok · ${self.cost_usd:.4f} · {self.requests} req "
        )

    def turn_summary(self, delta: UsageSnapshot) -> str:
        return (
            f"[usage] turn {_fmt(delta.input_tokens)} in / {_fmt(delta.output_tokens)} out"
            f" (${delta.cost_usd:.4f}) · session {_fmt(self.total_tokens)} tok ${self.cost_usd:.4f}"
        )

    def breakdown(self) -> str:
        return "\n".join(
            [
                f"Model:          {self.model}",
                f"Context (last): {self.context_text()}",
                f"Requests:       {self.requests}",
                f"Input tokens:   {self.input_tokens:,} (cached {self.cached_input_tokens:,})",
                f"Output tokens:  {self.output_tokens:,} (reasoning {self.reasoning_tokens:,})",
                f"Total tokens:   {self.total_tokens:,}",
                f"Estimated cost: ${self.cost_usd:.4f}",
            ]
        )


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _detail(usage: Any, group: str, field: str) -> int:
    return _int(getattr(getattr(usage, group, None), field, 0))


def _fmt(tokens: int) -> str:
    if tokens < 1000:
        return str(tokens)
    return f"{tokens / 1000:.1f}k".replace(".0k", "k")
