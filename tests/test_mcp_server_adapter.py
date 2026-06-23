from __future__ import annotations

import unittest
from typing import Any

import diffbot_agent.openai_agents_runtime as runtime
from diffbot_agent.openai_agents_runtime import (
    _elicitation_mcp_server_class,
    _mcp_server_class_and_kwargs,
)


class McpServerAdapterTest(unittest.TestCase):
    def test_uses_native_elicitation_callback_when_available(self) -> None:
        class NativeServer:
            def __init__(self, *, elicitation_callback: Any | None = None) -> None:
                self.elicitation_callback = elicitation_callback

        callback = object()
        server_class, kwargs = _mcp_server_class_and_kwargs(NativeServer, callback)

        self.assertIs(server_class, NativeServer)
        self.assertEqual(kwargs, {"elicitation_callback": callback})

    def test_fallback_adapter_is_not_the_logging_wrapper(self) -> None:
        class PlainServer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        adapter = _elicitation_mcp_server_class(PlainServer)

        self.assertEqual(adapter.__name__, "ElicitationMCPServerStreamableHttp")
        self.assertNotIn("list_tools", adapter.__dict__)
        self.assertNotIn("call_tool", adapter.__dict__)
        self.assertFalse(hasattr(runtime, "_logging_mcp_server_class"))


if __name__ == "__main__":
    unittest.main()
