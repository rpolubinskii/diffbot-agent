from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from diffbot_agent.episode import build_canonical_record
from diffbot_agent.memory_backend import (
    NullMemoryBackend,
    SqliteRecencyBackend,
    clear_command_memories,
)


def _record(session_id: str, command: str, *, completed_at: str) -> object:
    return build_canonical_record(
        session_id=session_id,
        started_at="2026-06-22T11:59:00+00:00",
        completed_at=completed_at,
        command=command,
        completion_status="completed",
        items=[
            {"type": "function_call", "call_id": "n1", "name": "x__navigate_to", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "n1", "output": '{"success":true}'},
        ],
        final_output="done",
    )


class SqliteRecencyBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = str(Path(tempfile.mkdtemp()) / "mem.sqlite3")
        self.now = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)

    def test_recall_empty_is_none(self) -> None:
        backend = SqliteRecencyBackend(self.db, "sess")
        try:
            text = asyncio.run(backend.recall(query="anything", limit=4, now=self.now))
            self.assertEqual(text, "(none)")
        finally:
            asyncio.run(backend.close())

    def test_add_then_recall_and_reset(self) -> None:
        backend = SqliteRecencyBackend(self.db, "sess")
        try:
            asyncio.run(backend.add_episode(_record("sess", "drive forward", completed_at="2026-06-22T11:59:30+00:00")))
            text = asyncio.run(backend.recall(query="x", limit=4, now=self.now))
            self.assertIn("drive forward", text)
            asyncio.run(backend.reset())
            self.assertEqual(asyncio.run(backend.recall(query="x", limit=4, now=self.now)), "(none)")
        finally:
            asyncio.run(backend.close())

    def test_recall_is_session_scoped_and_limited(self) -> None:
        backend = SqliteRecencyBackend(self.db, "sess")
        try:
            asyncio.run(backend.add_episode(_record("sess", "first", completed_at="2026-06-22T11:59:10+00:00")))
            asyncio.run(backend.add_episode(_record("sess", "second", completed_at="2026-06-22T11:59:20+00:00")))
            asyncio.run(backend.add_episode(_record("other", "elsewhere", completed_at="2026-06-22T11:59:25+00:00")))
            text = asyncio.run(backend.recall(query="x", limit=1, now=self.now))
            self.assertIn("second", text)
            self.assertNotIn("first", text)
            self.assertNotIn("elsewhere", text)
        finally:
            asyncio.run(backend.close())

    def test_clear_command_memories_helper(self) -> None:
        backend = SqliteRecencyBackend(self.db, "sess")
        asyncio.run(backend.add_episode(_record("sess", "drive forward", completed_at="2026-06-22T11:59:30+00:00")))
        asyncio.run(backend.close())
        clear_command_memories(self.db, "sess")
        reopened = SqliteRecencyBackend(self.db, "sess")
        try:
            self.assertEqual(asyncio.run(reopened.recall(query="x", limit=4, now=self.now)), "(none)")
        finally:
            asyncio.run(reopened.close())


class NullMemoryBackendTest(unittest.TestCase):
    def test_noop_behaviour(self) -> None:
        backend = NullMemoryBackend()
        now = datetime(2026, 6, 22, 12, 0, 0, tzinfo=timezone.utc)
        asyncio.run(backend.add_episode(_record("sess", "drive", completed_at="2026-06-22T11:59:30+00:00")))
        self.assertEqual(asyncio.run(backend.recall(query="x", limit=4, now=now)), "(none)")
        asyncio.run(backend.reset())
        asyncio.run(backend.close())


if __name__ == "__main__":
    unittest.main()
