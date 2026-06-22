from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from diffbot_agent.episode import CanonicalCommandRecord, render_recent_memories


EMPTY_MEMORY = "(none)"


@runtime_checkable
class MemoryBackend(Protocol):
    """Cross-command memory seam: default is SQLite recency; a future
    ``DiffbotRagBackend`` (Graphiti behind ``diffbot-rag``) drops in unchanged."""

    async def add_episode(self, record: CanonicalCommandRecord) -> None: ...

    async def recall(self, *, query: str, limit: int, now: datetime) -> str:
        """Return a rendered historical-memory block, ``"(none)"`` if empty."""

    async def reset(self) -> None: ...

    async def close(self) -> None: ...


class NullMemoryBackend:
    """No-op backend used when memory is disabled."""

    async def add_episode(self, record: CanonicalCommandRecord) -> None:
        return None

    async def recall(self, *, query: str, limit: int, now: datetime) -> str:
        return EMPTY_MEMORY

    async def reset(self) -> None:
        return None

    async def close(self) -> None:
        return None


class SqliteRecencyBackend:
    """Recency-based memory over local SQLite (the default backend)."""

    def __init__(self, db_path: str | Path, session_id: str):
        self.session_id = session_id
        self._store = CommandMemoryStore(db_path)

    async def add_episode(self, record: CanonicalCommandRecord) -> None:
        await self._store.add(record)

    async def recall(self, *, query: str, limit: int, now: datetime) -> str:
        records = await self._store.latest(self.session_id, limit)
        return render_recent_memories(records, now=now)

    async def reset(self) -> None:
        await self._store.clear(self.session_id)

    async def close(self) -> None:
        await self._store.close()


class CommandMemoryStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            _initialize_command_memory_table(self._connection)

    async def add(self, record: CanonicalCommandRecord) -> None:
        self._add_sync(record)

    def _add_sync(self, record: CanonicalCommandRecord) -> None:
        with self._lock:
            self._require_open()
            self._connection.execute(
                """
                INSERT INTO command_memories (
                    session_id,
                    started_at,
                    completed_at,
                    command,
                    completion_status,
                    final_assistant_text,
                    spoken_text,
                    tool_events,
                    navigation_outcomes,
                    safety_outcomes,
                    error_outcomes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.started_at,
                    record.completed_at,
                    record.command,
                    record.completion_status,
                    record.final_assistant_text,
                    _json_dump(list(record.spoken_text)),
                    _json_dump(list(record.tool_events)),
                    _json_dump(list(record.navigation_outcomes)),
                    _json_dump(list(record.safety_outcomes)),
                    _json_dump(list(record.error_outcomes)),
                ),
            )
            self._connection.commit()

    async def latest(
        self,
        session_id: str,
        limit: int,
    ) -> list[CanonicalCommandRecord]:
        if limit <= 0:
            return []
        return self._latest_sync(session_id, limit)

    def _latest_sync(
        self,
        session_id: str,
        limit: int,
    ) -> list[CanonicalCommandRecord]:
        with self._lock:
            self._require_open()
            rows = self._connection.execute(
                """
                SELECT
                    id,
                    session_id,
                    started_at,
                    completed_at,
                    command,
                    completion_status,
                    final_assistant_text,
                    spoken_text,
                    tool_events,
                    navigation_outcomes,
                    safety_outcomes,
                    error_outcomes
                FROM command_memories
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [_record_from_row(row) for row in reversed(rows)]

    async def clear(self, session_id: str) -> None:
        with self._lock:
            self._require_open()
            self._connection.execute(
                "DELETE FROM command_memories WHERE session_id = ?",
                (session_id,),
            )
            self._connection.commit()

    async def close(self) -> None:
        self._close_sync()

    def _close_sync(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._connection.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CommandMemoryStore is closed.")


def clear_command_memories(db_path: str | Path, session_id: str) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        _initialize_command_memory_table(connection)
        connection.execute(
            "DELETE FROM command_memories WHERE session_id = ?",
            (session_id,),
        )
        connection.commit()
    finally:
        connection.close()


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _initialize_command_memory_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS command_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            command TEXT NOT NULL,
            completion_status TEXT NOT NULL,
            final_assistant_text TEXT NOT NULL,
            spoken_text TEXT NOT NULL,
            tool_events TEXT NOT NULL,
            navigation_outcomes TEXT NOT NULL,
            safety_outcomes TEXT NOT NULL,
            error_outcomes TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_command_memories_session_id
        ON command_memories (session_id, id)
        """
    )
    connection.commit()


def _record_from_row(row: tuple[object, ...]) -> CanonicalCommandRecord:
    return CanonicalCommandRecord(
        record_id=int(row[0]),
        session_id=str(row[1]),
        started_at=str(row[2]),
        completed_at=str(row[3]),
        command=str(row[4]),
        completion_status=str(row[5]),
        final_assistant_text=str(row[6]),
        spoken_text=tuple(json.loads(row[7])),
        tool_events=tuple(json.loads(row[8])),
        navigation_outcomes=tuple(json.loads(row[9])),
        safety_outcomes=tuple(json.loads(row[10])),
        error_outcomes=tuple(json.loads(row[11])),
    )
