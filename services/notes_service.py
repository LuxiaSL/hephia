"""
services/notes_service.py

Standalone SQLite + FTS5 notes CRUD service.
Ported from brain/environments/notes.py — stripped of command infrastructure,
BaseEnvironment, and CommandDefinition parsing.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from shared_models.api_models import NoteResponse
from loggers import SystemLogger


class NotesService:
    """Persistent note storage with full-text search via FTS5."""

    def __init__(self, db_path: str = "data/notes.db", max_sticky: int = 3) -> None:
        self.db_path = db_path
        self.max_sticky = max_sticky
        self._ensure_db()

    # ---- Public API ----

    def create(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        sticky: bool = False,
        context: Optional[str] = None,
    ) -> NoteResponse:
        """Create a new note."""
        tags = tags or []
        conn = self._connect()
        try:
            if sticky:
                count = self._sticky_count(conn)
                if count >= self.max_sticky:
                    raise ValueError(
                        f"Cannot have more than {self.max_sticky} sticky notes. "
                        "Remove sticky from another note first."
                    )

            note_id = str(uuid4())
            now = datetime.utcnow().isoformat()
            tags_str = ",".join(tags) if tags else ""

            conn.execute(
                """INSERT INTO notes (id, content, created_at, updated_at, context, sticky, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (note_id, content, now, now, context, int(sticky), tags_str),
            )
            self._sync_tags(conn, note_id, tags)
            conn.commit()

            return NoteResponse(
                id=note_id,
                content=content,
                created_at=now,
                updated_at=now,
                context=context,
                sticky=sticky,
                tags=tags,
            )
        finally:
            conn.close()

    def get(self, note_id: str) -> Optional[NoteResponse]:
        """Get a single note by id."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, content, created_at, updated_at, context, sticky, tags FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_response(row)
        finally:
            conn.close()

    def update(
        self,
        note_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sticky: Optional[bool] = None,
    ) -> Optional[NoteResponse]:
        """Update an existing note. Only non-None fields are changed."""
        conn = self._connect()
        try:
            existing = conn.execute(
                "SELECT id, content, created_at, updated_at, context, sticky, tags FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
            if existing is None:
                return None

            current = self._row_to_response(existing)
            new_content = content if content is not None else current.content
            new_sticky = sticky if sticky is not None else current.sticky
            new_tags = tags if tags is not None else current.tags
            now = datetime.utcnow().isoformat()
            tags_str = ",".join(new_tags) if new_tags else ""

            if new_sticky and not current.sticky:
                count = self._sticky_count(conn)
                if count >= self.max_sticky:
                    raise ValueError(
                        f"Cannot have more than {self.max_sticky} sticky notes."
                    )

            conn.execute(
                """UPDATE notes SET content = ?, updated_at = ?, sticky = ?, tags = ?
                   WHERE id = ?""",
                (new_content, now, int(new_sticky), tags_str, note_id),
            )

            if tags is not None:
                self._sync_tags(conn, note_id, new_tags)

            conn.commit()

            return NoteResponse(
                id=note_id,
                content=new_content,
                created_at=current.created_at,
                updated_at=now,
                context=current.context,
                sticky=new_sticky,
                tags=new_tags,
            )
        finally:
            conn.close()

    def delete(self, note_id: str) -> bool:
        """Delete a note. Returns True if a note was deleted."""
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            if cursor.rowcount == 0:
                return False
            conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def list_notes(
        self,
        tag: Optional[str] = None,
        sticky_only: bool = False,
        limit: int = 20,
        offset: int = 0,
    ) -> List[NoteResponse]:
        """List notes with optional tag and sticky filters."""
        conn = self._connect()
        try:
            query = "SELECT n.id, n.content, n.created_at, n.updated_at, n.context, n.sticky, n.tags FROM notes n"
            params: list = []
            conditions: list = []

            if tag:
                query += " JOIN note_tags nt ON n.id = nt.note_id JOIN tags t ON nt.tag_id = t.id"
                conditions.append("t.name = ?")
                params.append(tag)

            if sticky_only:
                conditions.append("n.sticky = 1")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY n.updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_response(row) for row in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 10) -> List[NoteResponse]:
        """Full-text search across note content and tags."""
        conn = self._connect()
        try:
            fts_query = query.replace('"', '""')
            rows = conn.execute(
                """SELECT n.id, n.content, n.created_at, n.updated_at, n.context, n.sticky, n.tags
                   FROM notes n
                   JOIN notes_fts fts ON n.rowid = fts.rowid
                   WHERE notes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
            return [self._row_to_response(row) for row in rows]
        finally:
            conn.close()

    def list_tags(self) -> Dict[str, int]:
        """List all tags with their usage counts."""
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT t.name, COUNT(nt.note_id) as cnt
                   FROM tags t
                   JOIN note_tags nt ON t.id = nt.tag_id
                   GROUP BY t.name
                   ORDER BY cnt DESC""",
            ).fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            conn.close()

    def get_sticky_notes(self) -> List[NoteResponse]:
        """Get all sticky notes."""
        return self.list_notes(sticky_only=True, limit=self.max_sticky)

    # ---- Internals ----

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _sticky_count(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT COUNT(*) FROM notes WHERE sticky = 1").fetchone()
        return row[0] if row else 0

    def _sync_tags(self, conn: sqlite3.Connection, note_id: str, tags: List[str]) -> None:
        """Sync the note_tags junction table for a note."""
        conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
        for tag_name in tags:
            tag_name = tag_name.strip()
            if not tag_name:
                continue
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag_name,))
            tag_row = conn.execute("SELECT id FROM tags WHERE name = ?", (tag_name,)).fetchone()
            if tag_row:
                conn.execute(
                    "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                    (note_id, tag_row[0]),
                )

    def _row_to_response(self, row: tuple) -> NoteResponse:
        """Convert a database row to NoteResponse."""
        id_, content, created_at, updated_at, context, sticky, tags_str = row
        tags = [t.strip() for t in (tags_str or "").split(",") if t.strip()]
        return NoteResponse(
            id=id_,
            content=content,
            created_at=created_at or "",
            updated_at=updated_at or "",
            context=context,
            sticky=bool(sticky),
            tags=tags,
        )

    def _ensure_db(self) -> None:
        """Create tables, FTS, and triggers if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                sticky INTEGER DEFAULT 0,
                tags TEXT
            );

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS note_tags (
                note_id TEXT,
                tag_id INTEGER,
                FOREIGN KEY (note_id) REFERENCES notes(id),
                FOREIGN KEY (tag_id) REFERENCES tags(id),
                PRIMARY KEY (note_id, tag_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content,
                tags
            );
        """)

        # Ensure triggers exist for FTS sync
        cursor.executescript("""
            DROP TRIGGER IF EXISTS notes_after_insert;
            DROP TRIGGER IF EXISTS notes_after_update;
            DROP TRIGGER IF EXISTS notes_after_delete;

            CREATE TRIGGER notes_after_insert AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, content, tags)
                VALUES (new.rowid, new.content, new.tags);
            END;

            CREATE TRIGGER notes_after_update AFTER UPDATE ON notes BEGIN
                UPDATE notes_fts
                SET content = new.content, tags = new.tags
                WHERE rowid = new.rowid;
            END;

            CREATE TRIGGER notes_after_delete AFTER DELETE ON notes BEGIN
                DELETE FROM notes_fts WHERE rowid = old.rowid;
            END;
        """)

        # Create index on sticky for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_sticky ON notes(sticky)")

        conn.commit()
        conn.close()
