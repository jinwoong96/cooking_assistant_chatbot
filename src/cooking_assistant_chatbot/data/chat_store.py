"""Saved chat conversations, so the UI can keep several separate chats
(like ChatGPT's sidebar) that survive restarts and are the same list on
every device that opens the app.

Each conversation's messages are the only history the agent sees for it —
`app.run_turn` reads history from here by conversation id, not from what
the browser happens to be showing, so chats can't leak into each other.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    recipe_seq TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_messages_conv
    ON conversation_messages(conversation_id, id);
"""

_TITLE_MAX_CHARS = 30


@dataclass
class Conversation:
    id: int
    title: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    migrate(conn)


def migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
    if "recipe_seq" not in columns:
        conn.execute("ALTER TABLE conversations ADD COLUMN recipe_seq TEXT")
        conn.commit()


def set_recipe(conn: sqlite3.Connection, conversation_id: int, recipe_seq: str) -> None:
    """Remember the recipe this chat last showed, for cooking mode."""
    conn.execute(
        "UPDATE conversations SET recipe_seq = ? WHERE id = ?", (recipe_seq, conversation_id)
    )
    conn.commit()


def get_recipe_seq(conn: sqlite3.Connection, conversation_id: int | None) -> str | None:
    if conversation_id is None:
        return None
    row = conn.execute(
        "SELECT recipe_seq FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    return row[0] if row else None


def title_from_message(message: str) -> str:
    """The first user message, trimmed — no LLM call just to name a chat."""
    title = " ".join(message.split())
    if len(title) > _TITLE_MAX_CHARS:
        title = title[: _TITLE_MAX_CHARS - 1].rstrip() + "…"
    return title or "새 채팅"


def create_conversation(conn: sqlite3.Connection, title: str) -> int:
    now = _now()
    cursor = conn.execute(
        "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
        (title, now, now),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_conversations(conn: sqlite3.Connection) -> list[Conversation]:
    """Most recently used first."""
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
    ).fetchall()
    return [Conversation(id=row[0], title=row[1], updated_at=row[2]) for row in rows]


def get_messages(conn: sqlite3.Connection, conversation_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT role, content FROM conversation_messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [{"role": row[0], "content": row[1]} for row in rows]


def append_messages(
    conn: sqlite3.Connection, conversation_id: int, messages: list[dict]
) -> None:
    conn.executemany(
        "INSERT INTO conversation_messages (conversation_id, role, content) VALUES (?, ?, ?)",
        [(conversation_id, m["role"], m["content"]) for m in messages],
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
    )
    conn.commit()


def delete_conversation(conn: sqlite3.Connection, conversation_id: int) -> None:
    # Explicit, rather than relying on ON DELETE CASCADE: SQLite only
    # enforces foreign keys when PRAGMA foreign_keys is on per connection.
    conn.execute(
        "DELETE FROM conversation_messages WHERE conversation_id = ?", (conversation_id,)
    )
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
