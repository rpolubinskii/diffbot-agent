from __future__ import annotations

import json
import time
from typing import Any, cast

from agents import _debug
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem
from agents.logger import logger
from agents.models.chatcmpl_converter import Converter
from agents.models.chatcmpl_helpers import ChatCmplHelpers
from agents.models.fake_id import FAKE_RESPONSES_ID
from agents.models.interface import Model, ModelTracing
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.models.openai_provider import OpenAIProvider, get_default_model
from agents.models.openai_responses import Converter as OpenAIResponsesConverter
from agents.tool import Tool
from agents.tracing.span_data import GenerationSpanData
from agents.tracing.spans import Span
from agents.util._json import _to_dump_compatible
from openai import AsyncStream, omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response
from openai.types.responses.response_prompt_param import ResponsePromptParam

try:
    from agents.model_settings import ModelSettings
except ImportError:  # pragma: no cover - only needed for runtime type introspection.
    ModelSettings = Any  # type: ignore[assignment, misc]


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
    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Span[GenerationSpanData],
        tracing: ModelTracing,
        stream: bool = False,
        prompt: ResponsePromptParam | None = None,
    ) -> ChatCompletion | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        self._handle_unsupported_prompt(prompt)
        self._validate_official_openai_input_content_types(input)
        converted_messages = Converter.items_to_messages(
            input,
            model=self.model,
            base_url=str(self._client.base_url),
            should_replay_reasoning_content=self.should_replay_reasoning_content,
            strict_feature_validation=self._strict_feature_validation,
            preserve_tool_output_all_content=True,
        )

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if tracing.include_data():
            span.span_data.input = converted_messages

        if model_settings.parallel_tool_calls and tools:
            parallel_tool_calls: bool | Any = True
        elif model_settings.parallel_tool_calls is False:
            parallel_tool_calls = False
        else:
            parallel_tool_calls = omit
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []

        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))

        converted_tools = _to_dump_compatible(converted_tools)
        tools_param = converted_tools if converted_tools else omit

        if _debug.DONT_LOG_MODEL_DATA:
            logger.debug("Calling LLM")
        else:
            messages_json = json.dumps(
                converted_messages,
                indent=2,
                ensure_ascii=False,
            )
            tools_json = json.dumps(
                converted_tools,
                indent=2,
                ensure_ascii=False,
            )
            logger.debug(
                f"{messages_json}\n"
                f"Tools:\n{tools_json}\n"
                f"Stream: {stream}\n"
                f"Tool choice: {tool_choice}\n"
                f"Response format: {response_format}\n"
            )

        reasoning_effort = model_settings.reasoning.effort if model_settings.reasoning else None
        store = ChatCmplHelpers.get_store_param(self._get_client(), model_settings)

        stream_options = ChatCmplHelpers.get_stream_options_param(
            self._get_client(), model_settings, stream=stream
        )

        stream_param: bool | Any = True if stream else omit

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "tools": tools_param,
            "temperature": self._non_null_or_omit(model_settings.temperature),
            "top_p": self._non_null_or_omit(model_settings.top_p),
            "frequency_penalty": self._non_null_or_omit(model_settings.frequency_penalty),
            "presence_penalty": self._non_null_or_omit(model_settings.presence_penalty),
            "max_tokens": self._non_null_or_omit(model_settings.max_tokens),
            "tool_choice": tool_choice,
            "response_format": response_format,
            "parallel_tool_calls": parallel_tool_calls,
            "stream": cast(Any, stream_param),
            "stream_options": self._non_null_or_omit(stream_options),
            "store": self._non_null_or_omit(store),
            "reasoning_effort": self._non_null_or_omit(reasoning_effort),
            "verbosity": self._non_null_or_omit(model_settings.verbosity),
            "top_logprobs": self._non_null_or_omit(model_settings.top_logprobs),
            "prompt_cache_retention": self._non_null_or_omit(model_settings.prompt_cache_retention),
            "extra_headers": self._merge_headers(model_settings),
            "extra_query": model_settings.extra_query,
            "extra_body": model_settings.extra_body,
            "metadata": self._non_null_or_omit(model_settings.metadata),
        }
        duplicate_extra_arg_keys = sorted(
            set(create_kwargs).intersection(model_settings.extra_args or {})
        )
        if duplicate_extra_arg_keys:
            if len(duplicate_extra_arg_keys) == 1:
                key = duplicate_extra_arg_keys[0]
                raise TypeError(
                    f"chat.completions.create() got multiple values for keyword argument '{key}'"
                )
            keys = ", ".join(repr(key) for key in duplicate_extra_arg_keys)
            raise TypeError(
                f"chat.completions.create() got multiple values for keyword arguments {keys}"
            )
        create_kwargs.update(model_settings.extra_args or {})

        ret = await self._get_client().chat.completions.create(**create_kwargs)

        if isinstance(ret, ChatCompletion):
            return ret

        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            model_settings.tool_choice
        )
        if responses_tool_choice is None or responses_tool_choice is omit:
            responses_tool_choice = "auto"

        response = Response(
            id=FAKE_RESPONSES_ID,
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret
