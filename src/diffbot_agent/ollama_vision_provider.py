from __future__ import annotations

from typing import Any

from agents.models.chatcmpl_converter import Converter
from agents.models.interface import Model
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider, get_default_model


class OllamaVisionProvider(OpenAIProvider):
    """OpenAI-compatible provider that keeps image tool outputs for Ollama."""

    def get_model(self, model_name: str | None) -> Model:
        if self._use_responses:
            return super().get_model(model_name)

        resolved_model_name = model_name if model_name is not None else get_default_model()
        return OllamaVisionChatCompletionsModel(
            model=resolved_model_name,
            openai_client=self._get_client(),
            strict_feature_validation=self._strict_feature_validation,
        )


class OllamaVisionChatCompletionsModel(OpenAIChatCompletionsModel):
    """Chat-completions model that preserves image content in tool outputs.

    Ollama vision models need image blocks inside tool outputs to survive the
    Responses->chat-completions conversion. The base model drops them because
    ``Converter.items_to_messages`` defaults ``preserve_tool_output_all_content``
    to ``False``. Rather than copy the entire request builder (which rots on every
    SDK upgrade), scope that single converter flag around the inherited
    ``_fetch_response`` so all request-building changes flow through from the SDK.

    The only coupling is ``Converter.items_to_messages`` accepting that keyword;
    ``tests/test_ollama_vision_provider.py`` guards it so an SDK upgrade fails loudly.
    """

    async def _fetch_response(self, *args: Any, **kwargs: Any) -> Any:
        bound_items_to_messages = Converter.items_to_messages
        saved = Converter.__dict__.get("items_to_messages")

        def _preserve_images(items: Any, **converter_kwargs: Any) -> Any:
            converter_kwargs.setdefault("preserve_tool_output_all_content", True)
            return bound_items_to_messages(items, **converter_kwargs)

        Converter.items_to_messages = staticmethod(_preserve_images)
        try:
            return await super()._fetch_response(*args, **kwargs)
        finally:
            if saved is not None:
                Converter.items_to_messages = saved
            else:  # pragma: no cover - defensive
                del Converter.items_to_messages
