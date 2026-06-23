from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from diffbot_agent.episode import (
    _normalize_tool_name,
    _tool_category,
    set_mcp_tool_categories,
)
from diffbot_agent.openai_agents_runtime import (
    TOOL_CATEGORY_META_KEY,
    _load_mcp_tool_categories,
)


class _FakeServer:
    def __init__(self, tools: list[object]) -> None:
        self._tools = tools

    async def list_tools(self) -> list[object]:
        return self._tools


class _BoomServer:
    async def list_tools(self) -> list[object]:
        raise RuntimeError("server down")


class LoadMcpToolCategoriesTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_mcp_tool_categories({})

    def test_reads_categories_from_tool_meta(self) -> None:
        tools = [
            SimpleNamespace(name="nav.move_to", meta={TOOL_CATEGORY_META_KEY: "navigation"}),
            SimpleNamespace(name="speak.say", meta={TOOL_CATEGORY_META_KEY: "speech"}),
            SimpleNamespace(name="vision.get_camera_image", meta={TOOL_CATEGORY_META_KEY: "vision"}),
            SimpleNamespace(name="nav.stop", meta={TOOL_CATEGORY_META_KEY: "safety"}),
            SimpleNamespace(name="no_meta_tool", meta=None),
        ]
        asyncio.run(_load_mcp_tool_categories(_FakeServer(tools)))

        self.assertEqual(_tool_category(_normalize_tool_name("diffbot-mcp__nav.move_to")), "navigation")
        self.assertEqual(_tool_category(_normalize_tool_name("diffbot-mcp__speak.say")), "speech")
        self.assertEqual(_tool_category(_normalize_tool_name("diffbot-mcp__vision.get_camera_image")), "vision")
        self.assertEqual(_tool_category(_normalize_tool_name("diffbot-mcp__nav.stop")), "safety")

    def test_list_tools_failure_is_non_fatal(self) -> None:
        # Must not raise; startup proceeds and classification falls back to heuristic.
        asyncio.run(_load_mcp_tool_categories(_BoomServer()))


if __name__ == "__main__":
    unittest.main()
