"""
Notes environment for Hephia's cognitive system.

Provides a persistent note-taking system with tagging and search capabilities.
Direct port of the original exOS notes system to Python, with added
context awareness for cognitive integration.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from uuid import uuid4

from .base_environment import BaseEnvironment
from event_dispatcher import global_event_dispatcher, Event
from .terminal_formatter import TerminalFormatter, CommandResponse, EnvironmentHelp

class NotesEnvironment(BaseEnvironment):
    """
    Notes management environment.
    Allows the cognitive system to store and retrieve thoughts and observations.
    """
    
    def __init__(self):
        """Initialize the notes environment."""
        self.db_path = 'data/notes.db'
        self._ensure_db()
    
    def _ensure_db(self):
        """Ensure database and tables exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                sentiment TEXT
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
                context,
                tags,
                sentiment,
                content=notes
            );
        """)
        
        # Create triggers for FTS updates
        cursor.executescript("""
            CREATE TRIGGER IF NOT EXISTS notes_after_insert AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(content, context, sentiment)
                VALUES (NEW.content, NEW.context, NEW.sentiment);
            END;

            CREATE TRIGGER IF NOT EXISTS notes_after_update AFTER UPDATE ON notes BEGIN
                UPDATE notes_fts 
                SET content = NEW.content, 
                    context = NEW.context,
                    sentiment = NEW.sentiment
                WHERE rowid = NEW.rowid;
            END;

            CREATE TRIGGER IF NOT EXISTS notes_after_delete AFTER DELETE ON notes BEGIN
                DELETE FROM notes_fts WHERE rowid = OLD.rowid;
            END;
        """)
        
        conn.commit()
        conn.close()
    
    def get_commands(self) -> List[Dict[str, str]]:
        """Get available notes commands."""
        return [
            {
                "name": "create",
                "description": "Create a new note with optional tags"
            },
            {
                "name": "list",
                "description": "List notes, optionally filtered by tags"
            },
            {
                "name": "read",
                "description": "Read a specific note by ID"
            },
            {
                "name": "update",
                "description": "Update an existing note"
            },
            {
                "name": "delete",
                "description": "Delete a note"
            },
            {
                "name": "search",
                "description": "Search notes by content and tags"
            },
            {
                "name": "tags",
                "description": "List all tags or manage note tags"
            }
        ]
    
    async def handle_command(self, command: str, context: Dict[str, Any]) -> Dict[str, str]:
        parts = command.split(maxsplit=1)
        action = parts[0]
        params = parts[1] if len(parts) > 1 else ""

        try:
            if action == "create":
                return await self._create_note(params, context)
            elif action == "list":
                return await self._list_notes(params)
            elif action == "read":
                return await self._read_note(params)
            elif action == "update":
                return await self._update_note(params, context)
            elif action == "delete":
                return await self._delete_note(params)
            elif action == "search":
                return await self._search_notes(params)
            elif action == "tags":
                return await self._handle_tags(params)
            else:
                return TerminalFormatter.format_error(f"Unknown notes command: {action}")
        except Exception as e:
            return TerminalFormatter.format_error(str(e))
    
    async def _create_note(self, params: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Create a new note with current cognitive context."""
        # Parse content and tags
        tags = []
        content = params
        
        # Extract tags if specified
        if "--tags" in params:
            content, tag_str = params.split("--tags", 1)
            tags = [t.strip() for t in tag_str.strip().split(',')]
        
        # Create note
        note_id = str(uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get current cognitive state for context
            note_context = {
                'mood': context.get('pet_state', {}).get('mood', {}),
                'needs': context.get('pet_state', {}).get('needs', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Insert note
            cursor.execute(
                """INSERT INTO notes (id, content, context) 
                   VALUES (?, ?, ?)""",
                (note_id, content.strip(), json.dumps(note_context))
            )
            
            # Handle tags
            for tag in tags:
                # Insert tag if new
                cursor.execute(
                    "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                    (tag,)
                )
                
                # Get tag id
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                tag_id = cursor.fetchone()[0]
                
                # Link tag to note
                cursor.execute(
                    "INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                    (note_id, tag_id)
                )
            
            conn.commit()
            
            # Notify system of new note
            global_event_dispatcher.dispatch_event(
                Event("notes:created", {
                    "note_id": note_id,
                    "content": content,
                    "tags": tags,
                    "context": note_context
                })
            )
            
            return CommandResponse(
                title="Note Created",
                content=f"Created note with ID: {note_id}\nContent: {content}\nTags: {', '.join(tags) if tags else 'none'}",
                suggested_commands=[
                    "'notes list' to view recent notes",
                    "'notes read {note_id}' to view this note",
                    "'notes tags add {note_id} tag1 tag2' to add more tags"
                ]
            )
            
        finally:
            conn.close()
    
    async def _list_notes(self, params: str) -> Dict[str, str]:        
        """List notes with optional tag filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT n.*, GROUP_CONCAT(t.name) as tags
                FROM notes n
                LEFT JOIN note_tags nt ON n.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
            """
            
            # Handle tag filtering
            if params and "--tag" in params:
                tag = params.split("--tag", 1)[1].strip()
                query += f" WHERE t.name = ?"
                cursor.execute(query + " GROUP BY n.id ORDER BY n.created_at DESC LIMIT 10", (tag,))
            else:
                cursor.execute(query + " GROUP BY n.id ORDER BY n.created_at DESC LIMIT 10")
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResponse(
                    title="No Notes Found",
                    content="You haven't created any notes yet. Use 'notes create \"Your note content\" --tags tag1,tag2' to get started!",
                    suggested_commands=["notes create \"My first note\" --tags example"]
                )
            
            # Format notes list
            content = "Recent Notes:\n\n"
            for note in notes:
                content += f"ID: {note[0]}\nCreated: {note[2]}\nContent: {note[1][:50]}...\nTags: [{note[-1] or 'none'}]\n\n"

            return CommandResponse(
                title="Notes List",
                content=content,
                suggested_commands=["notes read <note_id>", "notes search <query>"]
            )
            
        finally:
            conn.close()
    
    async def _read_note(self, note_id: str) -> Dict[str, str]:
        """Read a specific note."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT n.*, GROUP_CONCAT(t.name) as tags
                FROM notes n
                LEFT JOIN note_tags nt ON n.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
                WHERE n.id = ?
                GROUP BY n.id
            """, (note_id.strip(),))
            
            note = cursor.fetchone()
            if not note:
                return TerminalFormatter.format_error(f"Note {note_id} not found.")
            
            content = f"""Note {note[0]}:
Created: {note[2]}
Updated: {note[3]}
Tags: {note[-1] or 'none'}

Content:
{note[1]}"""

            return CommandResponse(
                title=f"Note {note_id}",
                content=content,
                suggested_commands=[
                    f"notes update {note_id} \"New content\" --tags tag1,tag2",
                    f"notes delete {note_id}"
                ]
            )
            
        finally:
            conn.close()
    
    async def _update_note(self, params: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Update an existing note."""
        parts = params.split(maxsplit=1)
        if len(parts) != 2:
            return "Usage: update <note_id> <new_content> [--tags tag1,tag2,...]"
            
        note_id, content = parts
        
        # Extract tags if specified
        tags = []
        if "--tags" in content:
            content, tag_str = content.split("--tags", 1)
            tags = [t.strip() for t in tag_str.strip().split(',')]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Update note
            cursor.execute(
                """UPDATE notes 
                   SET content = ?, 
                       updated_at = CURRENT_TIMESTAMP,
                       context = ?
                   WHERE id = ?""",
                (
                    content.strip(),
                    json.dumps({
                        'mood': context.get('pet_state', {}).get('mood', {}),
                        'needs': context.get('pet_state', {}).get('needs', {}),
                        'timestamp': datetime.now().isoformat()
                    }),
                    note_id
                )
            )
            
            if cursor.rowcount == 0:
                return f"Note {note_id} not found."
            
            # Update tags if specified
            if tags:
                # Remove old tags
                cursor.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
                
                # Add new tags
                for tag in tags:
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                        (tag,)
                    )
                    cursor.execute(
                        "SELECT id FROM tags WHERE name = ?",
                        (tag,)
                    )
                    tag_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                        (note_id, tag_id)
                    )
            
            conn.commit()
            
            return CommandResponse(
                title="Note Updated",
                content=f"Updated note {note_id}\nNew content: {content}\nNew tags: {', '.join(tags) if tags else 'unchanged'}",
                suggested_commands=[f"notes read {note_id}"],
                context=context
            )
            
        finally:
            conn.close()
    
    async def _delete_note(self, note_id: str) -> Dict[str, str]:
        """Delete a note."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id.strip(),))
            
            if cursor.rowcount == 0:
                return f"Note {note_id} not found."
                
            # Note tags are automatically deleted via foreign key constraints
            conn.commit()
            
            return CommandResponse(
                title="Note Deleted",
                content=f"Deleted note {note_id}",
                suggested_commands=["notes list"]
            )
            
        finally:
            conn.close()
    
    async def _search_notes(self, query: str) -> Dict[str, str]:
        """Search notes using full-text search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT notes.*, GROUP_CONCAT(t.name) as tags
                FROM notes_fts
                JOIN notes ON notes_fts.rowid = notes.rowid
                LEFT JOIN note_tags nt ON notes.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
                WHERE notes_fts MATCH ?
                GROUP BY notes.id
                ORDER BY rank
                LIMIT 5
            """, (query,))
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResponse(
                    title="No Matching Notes",
                    content=f"No notes found matching '{query}'",
                    suggested_commands=["notes list"]
                )
            
            content = f"Search results for '{query}':\n\n"
            for note in notes:
                content += f"ID: {note[0]}\nContent: {note[1][:50]}...\nTags: [{note[-1] or 'none'}]\n\n"

            return CommandResponse(
                title="Search Results",
                content=content,
                suggested_commands=["notes read <note_id>"]
            )
            
        finally:
            conn.close()
    
    async def _handle_tags(self, params: str) -> str:
        """Handle tag listing and management."""
        if not params:
            # List all tags
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT t.name, COUNT(nt.note_id) as count
                    FROM tags t
                    LEFT JOIN note_tags nt ON t.id = nt.tag_id
                    GROUP BY t.name
                    ORDER BY count DESC
                """)
                
                tags = cursor.fetchall()
                
                if not tags:
                    return "No tags found."
                
                response = "Available Tags:\n"
                for tag, count in tags:
                    response += f"{tag} ({count} note{'s' if count != 1 else ''})\n"
                    
                return response
                
            finally:
                conn.close()
        
        # Handle tag commands (add/remove)
        parts = params.split()
        if len(parts) < 3:
            return "Usage: tags add <note_id> <tag1> [tag2...] or tags remove <note_id> <tag1> [tag2...]"
            
        action, note_id, *tags = parts
        
        if action not in ['add', 'remove']:
            return f"Unknown tag action: {action}"
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if action == 'add':
                for tag in tags:
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                        (tag,)
                    )
                    cursor.execute(
                        "SELECT id FROM tags WHERE name = ?",
                        (tag,)
                    )
                    tag_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                        (note_id, tag_id)
                    )
            else:  # remove
                for tag in tags:
                    cursor.execute("""
                        DELETE FROM note_tags 
                        WHERE note_id = ? AND tag_id IN (
                            SELECT id FROM tags WHERE name = ?
                        )
                    """, (note_id, tag))
            
            conn.commit()
            
            return f"{action.capitalize()}ed tags {', '.join(tags)} {'to' if action == 'add' else 'from'} note {note_id}"
            
        finally:
            conn.close()

    def format_help(self) -> Dict[str, str]:
        return TerminalFormatter.format_environment_help(
            EnvironmentHelp(
                environment_name="Notes",
                commands=self.get_commands(),
                examples=[
                    "notes create \"My new note\" --tags tag1,tag2",
                    "notes list --tag important",
                    "notes search \"project ideas\""
                ],
                tips=[
                    "Use tags to organize your notes",
                    "Regularly review and update your notes",
                    "Use search to find notes related to a specific topic"
                ]
            )
        )