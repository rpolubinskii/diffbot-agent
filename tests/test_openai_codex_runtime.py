from __future__ import annotations

import asyncio
import sqlite3
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

from diffbot_agent.command_memory import CanonicalCommandRecord, CommandMemoryStore
from diffbot_agent.main import reset_session
from diffbot_agent.openai_codex_runtime import (
    _exclude_session_history,
    _image_sanitizing_session_class,
)


def test_session_history_is_excluded() -> None:
    old = [{"role": "user", "content": "old"}]
    new = [{"role": "user", "content": "new"}]

    assert _exclude_session_history(old, new) == new


def test_image_sanitizing_session_copies_before_persistence() -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.items = None

        async def add_items(self, items):
            self.items = items

    async def exercise() -> None:
        session_class = _image_sanitizing_session_class(FakeSession)
        session = session_class()
        items = [
            {
                "type": "function_call_output",
                "call_id": "camera",
                "output": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,QUJD",
                    }
                ],
            }
        ]

        await session.add_items(items)

        assert "base64" in str(items)
        assert "base64" not in str(session.items)

    asyncio.run(exercise())


def test_reset_session_clears_sdk_and_canonical_history(tmp_path) -> None:
    async def exercise() -> None:
        db_path = tmp_path / "session.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.execute(
            "CREATE TABLE agent_sessions (session_id TEXT PRIMARY KEY)"
        )
        connection.execute(
            """
            CREATE TABLE agent_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_data TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO agent_sessions (session_id) VALUES ('session')"
        )
        connection.execute(
            """
            INSERT INTO agent_messages (session_id, message_data)
            VALUES ('session', '{"role":"user","content":"raw history"}')
            """
        )
        connection.commit()
        connection.close()

        store = CommandMemoryStore(db_path)
        await store.add(
            CanonicalCommandRecord(
                session_id="session",
                started_at="start",
                completed_at="complete",
                command="command",
                completion_status="completed",
                final_assistant_text="",
                spoken_text=(),
                tool_events=(),
                navigation_outcomes=(),
                safety_outcomes=(),
                error_outcomes=(),
                searchable_text="command",
            )
        )
        await store.close()

        config = SimpleNamespace(
            agent=SimpleNamespace(
                session_id="session",
                session_db=str(db_path),
            )
        )

        class FakeSQLiteSession:
            def __init__(self, session_id, session_db):
                self.session_id = session_id
                self.session_db = session_db

            async def clear_session(self):
                connection = sqlite3.connect(self.session_db)
                connection.execute(
                    "DELETE FROM agent_messages WHERE session_id = ?",
                    (self.session_id,),
                )
                connection.execute(
                    "DELETE FROM agent_sessions WHERE session_id = ?",
                    (self.session_id,),
                )
                connection.commit()
                connection.close()

            def close(self):
                return None

        fake_agents = ModuleType("agents")
        fake_agents.SQLiteSession = FakeSQLiteSession
        with patch.dict(sys.modules, {"agents": fake_agents}):
            await reset_session(config)

        connection = sqlite3.connect(db_path)
        assert connection.execute("SELECT * FROM agent_messages").fetchall() == []
        assert connection.execute("SELECT * FROM agent_sessions").fetchall() == []
        connection.close()
        store = CommandMemoryStore(db_path)
        assert await store.latest("session", 4) == []
        await store.close()

    asyncio.run(exercise())
