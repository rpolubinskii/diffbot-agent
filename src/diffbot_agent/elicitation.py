from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mcp import types

from diffbot_agent.logging_utils import log_event


DEFAULT_ELICITATION_TIMEOUT_SECONDS = 120.0

ElicitationAnswerProvider = Callable[[float], Awaitable[str | None]]


def build_elicitation_callback(answer_provider: ElicitationAnswerProvider):
    async def elicitation_callback(
        context: Any,
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | types.ErrorData:
        del context
        if not isinstance(params, types.ElicitRequestFormParams):
            return types.ErrorData(
                code=types.INVALID_REQUEST,
                message="Only form elicitation is supported.",
            )
        if not _is_raw_answer_schema(params.requestedSchema):
            return types.ErrorData(
                code=types.INVALID_REQUEST,
                message="Only a required string answer field is supported.",
            )

        timeout_seconds = _timeout_seconds(params.meta)
        log_event(
            "mcp.elicitation.request",
            {"message": params.message, "timeout_seconds": timeout_seconds},
        )
        answer = await answer_provider(timeout_seconds)
        if answer is None:
            log_event(
                "mcp.elicitation.cancel",
                {"message": params.message, "timeout_seconds": timeout_seconds},
                level=logging.WARNING,
            )
            return types.ElicitResult(action="cancel")

        log_event(
            "mcp.elicitation.accept",
            {"message": params.message, "answer_length": len(answer)},
        )
        return types.ElicitResult(action="accept", content={"answer": answer})

    return elicitation_callback


def _is_raw_answer_schema(schema: dict[str, Any]) -> bool:
    if schema.get("type") != "object":
        return False
    required = schema.get("required")
    if required != ["answer"]:
        return False
    properties = schema.get("properties")
    if not isinstance(properties, dict) or set(properties) != {"answer"}:
        return False
    answer = properties.get("answer")
    return isinstance(answer, dict) and answer.get("type") == "string"


def _timeout_seconds(meta: Any) -> float:
    if meta is None:
        return DEFAULT_ELICITATION_TIMEOUT_SECONDS
    if isinstance(meta, dict):
        value = meta.get("timeoutSeconds")
    else:
        value = getattr(meta, "timeoutSeconds", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return DEFAULT_ELICITATION_TIMEOUT_SECONDS
    return float(value)
