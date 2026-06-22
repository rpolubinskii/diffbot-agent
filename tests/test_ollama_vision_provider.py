from __future__ import annotations

import asyncio
import inspect
import unittest

from agents.models.chatcmpl_converter import Converter
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel

from diffbot_agent.ollama_vision_provider import (
    OllamaVisionChatCompletionsModel,
    OllamaVisionProvider,
)


class OllamaVisionProviderTest(unittest.TestCase):
    def test_sdk_converter_still_supports_preserve_flag(self) -> None:
        # Guards the SDK coupling: fail loudly if a pinned version drops this keyword.
        params = inspect.signature(Converter.items_to_messages).parameters
        self.assertIn("preserve_tool_output_all_content", params)

    def test_get_model_returns_vision_model_for_chat_completions(self) -> None:
        provider = OllamaVisionProvider(
            api_key="ollama",
            base_url="http://localhost:11434/v1",
            use_responses=False,
        )
        model = provider.get_model("qwen3")
        self.assertIsInstance(model, OllamaVisionChatCompletionsModel)

    def test_fetch_response_injects_preserve_flag_and_restores(self) -> None:
        calls: list[dict[str, object]] = []

        def spy(items: object, **kwargs: object) -> list[str]:
            calls.append(kwargs)
            return ["msg"]

        async def fake_super(self: object, *args: object, **kwargs: object) -> object:
            return Converter.items_to_messages([], model="qwen3")

        saved_converter = Converter.__dict__["items_to_messages"]
        saved_super = OpenAIChatCompletionsModel.__dict__["_fetch_response"]
        Converter.items_to_messages = staticmethod(spy)  # type: ignore[assignment]
        installed_spy = Converter.__dict__["items_to_messages"]
        OpenAIChatCompletionsModel._fetch_response = fake_super  # type: ignore[assignment]
        try:
            model = OllamaVisionChatCompletionsModel.__new__(OllamaVisionChatCompletionsModel)
            result = asyncio.run(
                model._fetch_response("sys", [], None, [], None, [], None, None)
            )
            self.assertEqual(result, ["msg"])
            self.assertEqual(len(calls), 1)
            self.assertTrue(calls[0].get("preserve_tool_output_all_content"))
            self.assertIs(Converter.__dict__["items_to_messages"], installed_spy)
        finally:
            Converter.items_to_messages = saved_converter  # type: ignore[assignment]
            OpenAIChatCompletionsModel._fetch_response = saved_super  # type: ignore[assignment]


if __name__ == "__main__":
    unittest.main()
