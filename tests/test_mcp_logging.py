from __future__ import annotations

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffbot_agent.mcp_logging import McpRunLogger


def _event(name: str, item: object) -> SimpleNamespace:
    return SimpleNamespace(type="run_item_stream_event", name=name, item=item)


def _mcp_origin() -> SimpleNamespace:
    return SimpleNamespace(type="mcp", mcp_server_name="diffbot-mcp")


class McpRunLoggerTest(unittest.TestCase):
    def test_tool_call_logs_request_and_stores_start(self) -> None:
        logger = McpRunLogger()
        item = SimpleNamespace(
            tool_origin=_mcp_origin(),
            raw_item=SimpleNamespace(
                name="mcp_diffbot-mcp__nav_get_pose",
                call_id="call_1",
                arguments='{"timeoutSeconds": 1}',
            ),
        )

        with (
            patch("diffbot_agent.mcp_logging.monotonic_ms", return_value=100.0),
            patch("diffbot_agent.mcp_logging.log_by_verbosity") as log_by_verbosity,
        ):
            logger.handle_event(_event("tool_called", item))

        log_by_verbosity.assert_called_once()
        self.assertEqual(log_by_verbosity.call_args.kwargs["debug_event"], "mcp.tool.request")
        self.assertEqual(
            log_by_verbosity.call_args.kwargs["debug_payload"],
            {
                "server": "diffbot-mcp",
                "tool": "mcp_diffbot-mcp__nav_get_pose",
                "call_id": "call_1",
                "arguments": {"timeoutSeconds": 1},
            },
        )

    def test_tool_output_logs_response_with_duration(self) -> None:
        logger = McpRunLogger()
        call_item = SimpleNamespace(
            tool_origin=_mcp_origin(),
            raw_item=SimpleNamespace(
                name="mcp_diffbot-mcp__nav_get_pose",
                call_id="call_1",
                arguments="{}",
            ),
        )
        output_item = SimpleNamespace(
            tool_origin=_mcp_origin(),
            raw_item=SimpleNamespace(call_id="call_1"),
            output={"ok": True},
        )

        with (
            patch("diffbot_agent.mcp_logging.monotonic_ms", return_value=100.0),
            patch("diffbot_agent.mcp_logging.elapsed_ms", return_value=25),
            patch("diffbot_agent.mcp_logging.log_by_verbosity"),
            patch("diffbot_agent.mcp_logging.log_event") as log_event,
        ):
            logger.handle_event(_event("tool_called", call_item))
            logger.handle_event(_event("tool_output", output_item))

        log_event.assert_called_once_with(
            "mcp.tool.response",
            {
                "server": "diffbot-mcp",
                "tool": "mcp_diffbot-mcp__nav_get_pose",
                "call_id": "call_1",
                "duration_ms": 25,
                "result": {"ok": True},
            },
        )

    def test_tool_output_error_marker_logs_warning(self) -> None:
        logger = McpRunLogger()
        output_item = SimpleNamespace(
            tool_origin=_mcp_origin(),
            raw_item=SimpleNamespace(call_id="call_1"),
            output={"isError": True},
        )

        with patch("diffbot_agent.mcp_logging.log_event") as log_event:
            logger.handle_event(_event("tool_output", output_item))

        self.assertEqual(log_event.call_args_list[1].args[0], "mcp.tool.result_error")
        self.assertEqual(log_event.call_args_list[1].kwargs["level"], logging.WARNING)

    def test_list_tools_event_logs_tool_names(self) -> None:
        logger = McpRunLogger()
        item = SimpleNamespace(
            raw_item=SimpleNamespace(
                server_label="diffbot-mcp",
                tools=[SimpleNamespace(name="nav.get_pose"), SimpleNamespace(name="speak.ask")],
                error=None,
            )
        )

        with patch("diffbot_agent.mcp_logging.log_event") as log_event:
            logger.handle_event(_event("mcp_list_tools", item))

        log_event.assert_called_once_with(
            "mcp.list_tools.response",
            {
                "server": "diffbot-mcp",
                "tool_count": 2,
                "tools": ["nav.get_pose", "speak.ask"],
                "error": None,
            },
        )

    def test_ignores_non_mcp_tool_events(self) -> None:
        logger = McpRunLogger()
        item = SimpleNamespace(
            tool_origin=SimpleNamespace(type="function"),
            raw_item=SimpleNamespace(name="local_tool", call_id="call_1", arguments="{}"),
        )

        with (
            patch("diffbot_agent.mcp_logging.log_by_verbosity") as log_by_verbosity,
            patch("diffbot_agent.mcp_logging.log_event") as log_event,
        ):
            logger.handle_event(_event("tool_called", item))

        log_by_verbosity.assert_not_called()
        log_event.assert_not_called()

    def test_pending_call_error_logs_transport_failure(self) -> None:
        logger = McpRunLogger()
        item = SimpleNamespace(
            tool_origin=_mcp_origin(),
            raw_item=SimpleNamespace(name="mcp_diffbot-mcp__nav_stop", call_id="call_1", arguments="{}"),
        )

        with (
            patch("diffbot_agent.mcp_logging.monotonic_ms", return_value=100.0),
            patch("diffbot_agent.mcp_logging.elapsed_ms", return_value=25),
            patch("diffbot_agent.mcp_logging.log_by_verbosity"),
            patch("diffbot_agent.mcp_logging.log_event") as log_event,
        ):
            logger.handle_event(_event("tool_called", item))
            logger.log_pending_errors(RuntimeError("server down"))

        log_event.assert_called_once_with(
            "mcp.tool.error",
            {
                "server": "diffbot-mcp",
                "tool": "mcp_diffbot-mcp__nav_stop",
                "call_id": "call_1",
                "duration_ms": 25,
                "error_type": "RuntimeError",
                "error": "server down",
            },
            level=logging.ERROR,
        )


if __name__ == "__main__":
    unittest.main()
