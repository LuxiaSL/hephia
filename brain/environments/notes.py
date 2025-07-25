"""
notes.py - Persistent note-taking environment for Hephia.

Provides a structured note management system with tagging and search capabilities.
Maintains an SQLite database for storage while exposing a clean command interface.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from uuid import uuid4

from config import Config
from .base_environment import BaseEnvironment
from event_dispatcher import global_event_dispatcher, Event
from brain.commands.model import (
    CommandDefinition, 
    Parameter, 
    Flag, 
    ParameterType,
    CommandResult,
    CommandValidationError
)

class NotesEnvironment(BaseEnvironment):
    """Notes environment for storing and managing cognitive observations."""

    def __init__(self):
        self.db_path = 'data/notes.db'
        self.max_sticky_notes = Config.MAX_STICKY_NOTES  # Configurable max sticky notes
        self._ensure_db()  # Setup database first
        super().__init__()  # Then initialize command structure
        
        self.help_text = f"""
        The notes environment provides persistent external storage for thoughts and observations.

        Core Features:
        - Create and manage notes with content and tags
        - Search notes by content or tags
        - Track note context and metadata
        - Organize with a tagging system

        Tips:
        - Use tags consistently for better organization
        - Include context in searches
        - Review and update notes regularly

        [NEW] Sticky Notes:
        - sticky notes can be used to pin whatever you'd like to your system prompt/HUD.
        - you can think of these as a way to set goals or persistent reminders, and can have up to {self.max_sticky_notes} sticky notes at a time.
        """

    def _register_commands(self) -> None:
        """Register all notes environment commands."""
        # Create command
        self.register_command(
            CommandDefinition(
                name="create",
                description="Create a new note with optional tags",
                parameters=[
                    Parameter(
                        name="content",
                        description="The note content",
                        required=True,
                        examples=['"My first note"', '"Important observation"']
                    )
                ],
                flags=[
                    Flag(
                        name="tags",
                        description="Comma-separated list of tags",
                        type=ParameterType.STRING,
                        required=False,
                        examples=["--tags=important,todo", "--tags=project,idea"]
                    ),
                    Flag(
                        name="sticky",
                        description="Make this note a sticky note",
                        type=ParameterType.BOOLEAN,
                        required=False,
                        examples=["--sticky=true", "--sticky=false"]
                    )
                ],
                examples=[
                    'notes create "<Content of note>"',
                    'notes create "<Content of note>" --tags=tags,go,here',
                    'notes create "<content to keep visible" --sticky=true'
                ],
                related_commands=['list', 'tags', 'sticky'],
                failure_hints={
                    "DatabaseError": "Failed to save note. Try again.",
                    "ValueError": "Invalid note format. Use quotes around content.",
                    "StickyLimitExceeded": f"Cannot have more than {self.max_sticky_notes} sticky notes. Remove sticky from another note first."
                }
            )
        )

        # List command
        self.register_command(
            CommandDefinition(
                name="list",
                description="List notes with optional filtering",
                parameters=[],
                flags=[
                    Flag(
                        name="tag",
                        description="Filter by specific tag",
                        type=ParameterType.STRING,
                        required=False,
                        examples=["--tag=important", "--tag=project"]
                    ),
                    Flag(
                        name="limit",
                        description="Maximum number of notes to show",
                        type=ParameterType.INTEGER,
                        default=10,
                        required=False,
                        examples=["--limit=5", "--limit=20"]
                    )
                ],
                examples=[
                    'notes list',
                    'notes list --tag=important',
                    'notes list --limit=5'
                ],
                related_commands=['create', 'search', 'tags'],
                failure_hints={
                    "DatabaseError": "Failed to retrieve notes. Try again."
                }
            )
        )

        # Read command
        self.register_command(
            CommandDefinition(
                name="read",
                description="Read a specific note by ID",
                parameters=[
                    Parameter(
                        name="note_id",
                        description="The ID of the note to read",
                        required=True,
                        examples=["note-1", "note-42"]
                    )
                ],
                examples=[
                    'notes read note-1',
                    'notes read note-42'
                ],
                related_commands=['update', 'delete'],
                failure_hints={
                    "NotFound": "Note not found. Check the ID and try again."
                }
            )
        )

        # Update command
        self.register_command(
            CommandDefinition(
                name="update",
                description="Update an existing note",
                parameters=[
                    Parameter(
                        name="note_id",
                        description="The ID of the note to update",
                        required=True,
                        examples=["note-1", "note-42"]
                    ),
                    Parameter(
                        name="content",
                        description="The new note content",
                        required=True,
                        examples=['"<new content>"']
                    )
                ],
                flags=[
                    Flag(
                        name="tags",
                        description="New comma-separated list of tags",
                        type=ParameterType.STRING,
                        required=False,
                        examples=["--tags=updated,important"]
                    )
                ],
                examples=[
                    'notes update note-1 "<new content>"',
                ],
                related_commands=['read', 'delete', 'sticky'],
                failure_hints={
                    "NotFound": "Note not found. Check the ID and try again.",
                    "ValueError": "Invalid update format. Check command syntax."
                }
            )
        )

        # Delete command
        self.register_command(
            CommandDefinition(
                name="delete",
                description="Delete a note",
                parameters=[
                    Parameter(
                        name="note_id",
                        description="The ID of the note to delete",
                        required=True,
                        examples=["note-1", "note-42"]
                    )
                ],
                examples=[
                    'notes delete note-1'
                ],
                related_commands=['read', 'update'],
                failure_hints={
                    "NotFound": "Note not found. Check the ID and try again."
                }
            )
        )

        # Search command
        self.register_command(
            CommandDefinition(
                name="search",
                description="Search notes by content and tags",
                parameters=[
                    Parameter(
                        name="query",
                        description="Search terms",
                        required=True,
                        examples=['"project ideas"', '"important tasks"']
                    )
                ],
                flags=[
                    Flag(
                        name="limit",
                        description="Maximum results to return",
                        type=ParameterType.INTEGER,
                        default=5,
                        required=False,
                        examples=["--limit=10"]
                    )
                ],
                examples=[
                    'notes search "project"',
                    'notes search "important" --limit=10'
                ],
                related_commands=['list', 'tags'],
                failure_hints={
                    "DatabaseError": "Search failed. Try simpler search terms."
                }
            )
        )

        # List tags command
        self.register_command(
            CommandDefinition(
                name="tags",
                description="List all used tags", 
                parameters=[],
                examples=[
                    'notes tags'
                ],
                failure_hints={
                    "DatabaseError": "Failed to retrieve tags. Try again."
                },
                related_commands=['list', 'search', 'tag-add', 'tag-remove']
            )
        )

        # Add tags command
        self.register_command(
            CommandDefinition(
                name="tag-add",
                description="Add tags to a note",
                parameters=[
                    Parameter(
                        name="note_id", 
                        description="Note ID to add tags to",
                        required=True,
                        examples=["note-1", "note-42"]
                    ),
                    Parameter(
                        name="tags",
                        description="Space-separated tags to add",
                        required=True,
                        examples=["important", "todo project"]
                    )
                ],
                examples=[
                    'notes tag-add note-1 important',
                    'notes tag-add note-42 project idea'
                ],
                failure_hints={
                    "NotFound": "Note not found. Check the ID.",
                    "ValueError": "Invalid tags format. Use space-separated tags.",
                    "DatabaseError": "Failed to add tags. Try again."
                },
                related_commands=['tags', 'tag-remove']
            )
        )

        # Remove tags command  
        self.register_command(
            CommandDefinition(
                name="tag-remove",
                description="Remove tags from a note",
                parameters=[
                    Parameter(
                        name="note_id",
                        description="Note ID to remove tags from",
                        required=True,
                        examples=["note-1", "note-42"]
                    ),
                    Parameter(
                        name="tags", 
                        description="Space-separated tags to remove",
                        required=True,
                        examples=["important", "todo project"]
                    )
                ],
                examples=[
                    'notes tag-remove note-1 important',
                    'notes tag-remove note-42 old-tag'
                ],
                failure_hints={
                    "NotFound": "Note not found. Check the ID.",
                    "ValueError": "Invalid tags format. Use space-separated tags.",
                    "DatabaseError": "Failed to remove tags. Try again." 
                },
                related_commands=['tags', 'tag-add']
            )
        )

        # Sticky notes command
        self.register_command(
            CommandDefinition(
                name="sticky",
                description=f"Toggle sticky status for a note (max {self.max_sticky_notes} sticky notes)",
                parameters=[
                    Parameter(
                        name="note_id",
                        description="The ID of the note to toggle sticky",
                        required=True,
                        examples=["note-1", "note-42"]
                    )
                ],
                examples=[
                    'notes sticky note-1',
                    'notes sticky note-42'
                ],
                related_commands=['list', 'read'],
                failure_hints={
                    "NotFound": "Note not found. Check the ID and try again.",
                    "StickyLimitExceeded": f"Cannot have more than {self.max_sticky_notes} sticky notes. Remove sticky from another note first."
                }
            )
        )
    
    def _ensure_db(self):
        """Ensure database and tables exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables.
        cursor.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,
                sentiment TEXT,
                tags TEXT  -- Denormalized tags field
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

            -- Note: Removed content=notes from the FTS table.
            CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
                content, 
                context,
                tags,
                sentiment
            );""")

        cursor.execute("PRAGMA table_info(notes)")
        existing_columns = {col[1] for col in cursor.fetchall()}
        
        # Add human_id column if missing
        if 'llm_id' not in existing_columns:
            # Add column without UNIQUE constraint first
            cursor.execute("ALTER TABLE notes ADD COLUMN llm_id TEXT")

            # Generate llm IDs for existing notes
            cursor.execute("SELECT id, rowid FROM notes WHERE llm_id IS NULL ORDER BY created_at")
            existing_notes = cursor.fetchall()
            for i, (note_id, rowid) in enumerate(existing_notes, 1):
                llm_id = f"note-{i}"
                cursor.execute("UPDATE notes SET llm_id = ? WHERE id = ?", (llm_id, note_id))

            # Now create unique index after data is populated
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_llm_id ON notes(llm_id)")

        # Add sticky column if missing
        if 'sticky' not in existing_columns:
            cursor.execute("ALTER TABLE notes ADD COLUMN sticky BOOLEAN DEFAULT 0")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_notes_sticky ON notes(sticky)")

        # Create triggers for full-text search (FTS) updates.
        cursor.executescript("""
            DROP TRIGGER IF EXISTS notes_after_insert;
            DROP TRIGGER IF EXISTS notes_after_update;
            DROP TRIGGER IF EXISTS notes_after_delete;
            
            CREATE TRIGGER notes_after_insert AFTER INSERT ON notes BEGIN
                INSERT INTO notes_fts(rowid, content, context, tags, sentiment)
                VALUES (new.rowid, new.content, new.context, new.tags, new.sentiment);
            END;

            CREATE TRIGGER notes_after_update AFTER UPDATE ON notes BEGIN
                UPDATE notes_fts 
                SET content = new.content, 
                    context = new.context,
                    tags = new.tags,
                    sentiment = new.sentiment
                WHERE rowid = new.rowid;
            END;

            CREATE TRIGGER notes_after_delete AFTER DELETE ON notes BEGIN
                DELETE FROM notes_fts WHERE rowid = old.rowid;
            END;
        """)
        
        conn.commit()
        conn.close()

    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """Execute a notes command with validated parameters."""
        try:
            if action == "create":
                return await self._create_note(params[0], context, flags.get("tags"), flags.get("sticky", False))
            elif action == "list":
                return await self._list_notes(
                    flags.get("tag"),
                    flags.get("limit", 10)
                )
            elif action == "read":
                return await self._read_note(params[0])
            elif action == "update":
                return await self._update_note(params[0], params[1], context, flags.get("tags"))
            elif action == "delete":
                return await self._delete_note(params[0])
            elif action == "search":
                return await self._search_notes(
                    params[0],
                    flags.get("limit", 5)
                )
            elif action == "tags":
                return await self._list_tags()
            elif action == "tag-add":
                return await self._handle_tags("add", params[0], *params[1].split())
            elif action == "tag-remove":
                return await self._handle_tags("remove", params[0], *params[1].split())
            elif action == "sticky":
                return await self._toggle_sticky(params[0])
            
            raise ValueError(f"Unknown action: {action}")
            
        except sqlite3.Error as e:
            return CommandResult(
                success=False,
                message=f"Database error: {str(e)}",
                error=CommandValidationError(
                    message=f"Database error: {str(e)}",
                    suggested_fixes=[
                        "Have admin check database connectivity and ensure no concurrency issues",
                        "Retry the operation"
                    ],
                    related_commands=["notes help"],
                    examples=["notes list --tag=important", 'notes create "Sample note"']
                ),
                suggested_commands=["notes help"]
            )
        except Exception as e:
            return CommandResult(
                success=False,
                message=str(e),
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=[
                        "Check your command syntax",
                        "Review the environment help: notes help"
                    ],
                    related_commands=["notes help"],
                    examples=["notes list", 'notes create "Sample note"']
                ),
                suggested_commands=["notes help"]
            )

    async def _create_note(
        self, 
        content: str, 
        context: Dict[str, Any],
        tags: Optional[str] = None,
        sticky: bool = False
    ) -> CommandResult:
        """
        Create a new note with context awareness.
        
        - Captures the current cognitive context.
        - Updates tag relationships and stores a denormalized tag string.
        - Notifies the system and suggests next actions.
        """
        note_id = str(uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            llm_id = self._get_next_llm_id(cursor)

            # Capture current cognitive state.
            note_context = {
                'mood': context.get('internal_state', {}).get('mood', {}),
                'needs': context.get('internal_state', {}).get('needs', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Prepare tags.
            tag_list = []
            tags_str = ""
            if tags:
                tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                tags_str = ",".join(tag_list)
            
            cursor.execute("SELECT COUNT(*) FROM notes WHERE sticky = 1")
            current_sticky_count = cursor.fetchone()[0]
            
            if current_sticky_count >= self.max_sticky_notes and sticky:
                return CommandResult(
                    success=False,
                    message=f"Cannot make note sticky: already have {current_sticky_count}/{self.max_sticky_notes} sticky notes",
                    suggested_commands=["notes list", f"notes sticky <note_id>  # to unstick another"],
                    error=CommandValidationError(
                        message=f"Sticky limit ({self.max_sticky_notes}) exceeded",
                        suggested_fixes=[
                            "Remove sticky from another note first",
                            "Check current sticky notes"
                        ],
                        related_commands=["notes list"],
                        examples=["notes sticky note-1  # to unstick", "notes list"]
                    )
                )

            # Store the note.
            cursor.execute(
                """INSERT INTO notes (id, llm_id, content, context, tags, sticky) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (note_id, llm_id, content.strip(), json.dumps(note_context), tags_str, sticky)
            )
            
            # Handle tag associations.
            if tag_list:
                for tag in tag_list:
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                        (tag,)
                    )
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                        (note_id, tag_id)
                    )
            
            conn.commit()
            
            # Notify system.
            global_event_dispatcher.dispatch_event(
                Event("notes:created", {
                    "note_id": note_id,
                    "llm_id": llm_id,
                    "content": content,
                    "tags": tag_list,
                    "context": note_context,
                    "sticky": bool(sticky)
                })
            )
            
            # Generate suggested next commands.
            suggested_next = [
                f"notes read {llm_id}",
                f"notes update {llm_id} \"<new content>\"",
            ]
            if tag_list:
                suggested_next.append(f"notes list --tag={tag_list[0]}")
            else:
                suggested_next.append(f'notes tag-add {llm_id} <tag>')

            to_display = f"{content[:75]}..." if len(content) > 75 else content
            
            return CommandResult(
                success=True,
                message=f"Created note {llm_id}\nContent: {to_display}\nTags: {', '.join(tag_list) if tag_list else 'none'}",
                suggested_commands=suggested_next,
                data={
                    "note_id": note_id,
                    "content": content,
                    "tags": tag_list,
                    "context": note_context,
                    "llm_id": llm_id,
                    "sticky": bool(sticky)
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to create note: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to create note: {str(e)}",
                    suggested_fixes=[
                        "Check input formatting and try again",
                        "Ensure the database is accessible"
                    ],
                    related_commands=["notes list", "notes help"],
                    examples=['notes create "My first note"', 'notes create "Another note" --tags=example']
                ),
                suggested_commands=[
                    'notes create "Try again"',
                    "notes list"
                ]
            )
        finally:
            conn.close()

    async def _read_note(self, note_identifier: str) -> CommandResult:
        """
        Read a specific note with its full context.
        Displays the note’s denormalized tags and parsed context.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            note_id = self._resolve_note_id(note_identifier, cursor)
            if not note_id:
                return CommandResult(
                    success=False,
                    message=f"Note {note_identifier} not found",
                    suggested_commands=[
                        "notes list",
                        'notes create "<new note>"'
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_identifier} not found",
                        suggested_fixes=[
                            "Check the note ID",
                            "Try 'notes list' to see available notes"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=["notes read <note_id>", "notes list --limit=5"]
                    )
                )
            
            cursor.execute("""
                SELECT n.id, n.content, n.created_at, n.updated_at, n.context, n.sentiment, 
                       n.llm_id, n.sticky, n.tags, GROUP_CONCAT(t.name) as tag_list
                FROM notes n
                LEFT JOIN note_tags nt ON n.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
                WHERE n.id = ?
                GROUP BY n.id
            """, (note_id,))
            
            note = cursor.fetchone()
            if not note:
                return CommandResult(
                    success=False,
                    message=f"Note {note_id} not found",
                    suggested_commands=[
                        "notes list",
                        'notes create "<new note>"'
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_id} not found",
                        suggested_fixes=[
                            "Check the note ID",
                            "Try 'notes list' to see available notes"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=["notes read <note_id>", "notes list --limit=5"]
                    )
                )
            
            # Parse stored context.
            context_data = json.loads(note[4]) if note[4] else {}
            
            # Format the note display.
            sticky_indicator = "📌 " if note[7] else ""
            content_lines = [
                f"{sticky_indicator}Note {note[6]}:",
                f"Created: {note[2]}",
                f"Updated: {note[3]}",
                f"Tags: {note[8] or 'none'}",
                "",
                "Content:",
                note[1]
            ]
            
            if context_data.get('mood'):
                content_lines.extend([
                    "",
                    "Created while:",
                    f"Mood: {context_data['mood'].get('name', 'unknown')}",
                    f"Needs: {', '.join(f'{k}: {v}' for k, v in context_data.get('needs', {}).items())}"
                ])

            suggested = [
                f"notes update {note[6]} \"New content\"",
                f"notes tag-add {note[6]} <tag>",
                f"notes sticky {note[6]}"
            ]
            if note[8]:
                tags = note[8].split(',')
                suggested.append(f"notes list --tag={tags[0]}")
            
            return CommandResult(
                success=True,
                message="\n".join(content_lines),
                suggested_commands=suggested,
                data={
                    "note_id": note_id,
                    "llm_id": note[6],
                    "content": note[1],
                    "created": note[2],
                    "updated": note[3],
                    "tags": note[8].split(',') if note[8] else [],
                    "sticky": bool(note[7]),
                    "context": context_data
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to read note: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to read note: {str(e)}",
                    suggested_fixes=[
                        "Verify the note ID exists",
                        "Try listing available notes first"
                    ],
                    related_commands=["notes list", "notes search"],
                    examples=[
                        "notes read note-1",
                        "notes list --limit=5",
                        'notes search "keyword"'
                    ]
                ),
                suggested_commands=[
                    "notes list", 
                    "notes search <query>",
                    'notes create "<new note>"'
                ]
            )
        finally:
            conn.close()

    async def _list_notes(self, tag: Optional[str] = None, limit: int = 10) -> CommandResult:
        """List notes with optional tag filtering."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            base_query = """
                SELECT n.llm_id, n.content, n.created_at, n.tags, n.sticky
                FROM notes n
                LEFT JOIN note_tags nt ON n.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
            """
            if tag:
                base_query += " WHERE t.name = ?"
                params = (tag, limit)
            else:
                params = (limit,)
                
            full_query = base_query + " GROUP BY n.id ORDER BY n.created_at DESC LIMIT ?"
            cursor.execute(full_query, params)
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResult(
                    success=True,
                    message="No notes found.",
                    suggested_commands=[
                        'notes create "My first note" --tags=example',
                        'notes create "Getting started"'
                    ],
                    data={"count": 0}
                )
            
            content_lines = ["Recent Notes:\n"]
            for llm_id, content, created, tags, sticky in notes:
                sticky_indicator = "📌 " if sticky else ""
                content_lines.append(
                    f"{sticky_indicator}ID: {llm_id}\n"
                    f"Created: {created}\n"
                    f"Content: {content[:75]}...\n"
                    f"Tags: {tags or 'none'}\n"
                )

            return CommandResult(
                success=True,
                message="\n".join(content_lines),
                suggested_commands=[
                    "notes read <note_id>",
                    "notes search <query>",
                    f"notes list --limit={limit + 5}"
                ],
                data={
                    "count": len(notes),
                    "has_more": len(notes) == limit,
                    "filter_tag": tag
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list notes: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to list notes: {str(e)}",
                    suggested_fixes=[
                        "Check database connectivity",
                        "Try with a smaller limit value",
                        "Verify tag name if filtering by tag"
                    ],
                    related_commands=["notes create", "notes search", "notes tags"],
                    examples=[
                        "notes list --limit=5",
                        "notes list --tag=important",
                        "notes list"
                    ]
                ),
                suggested_commands=[
                    'notes create "<new note>"',
                    "notes tags",
                    "notes search <query>"
                ]
            )
        finally:
            conn.close()
    
    async def _update_note(
        self,
        note_identifier: str,
        content: str,
        context: Dict[str, Any],
        tags: Optional[str] = None,
    ) -> CommandResult:
        """Update a note's content and metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Resolve note ID using the helper method
            note_id = self._resolve_note_id(note_identifier, cursor)
            if not note_id:
                return CommandResult(
                    success=False,
                    message=f"Note {note_identifier} not found",
                    suggested_commands=[
                        "notes list",
                        f'notes create "{content}"'
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_identifier} not found",
                        suggested_fixes=[
                            "Verify the note ID is correct",
                            "Use 'notes list' to see available note IDs"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=[f'notes update {note_identifier} "New content"', "notes list --limit=5"]
                    )
                )

            # Get the llm_id for the response
            cursor.execute("SELECT llm_id FROM notes WHERE id = ?", (note_id,))
            llm_id = cursor.fetchone()[0]

            new_context = {
                'mood': context.get('internal_state', {}).get('mood', {}),
                'needs': context.get('internal_state', {}).get('needs', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # If new tags are provided, process them; otherwise keep the existing denormalized tags.
            tags_str = None
            tag_list = []
            if tags is not None:
                tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                tags_str = ",".join(tag_list)
            else:
                cursor.execute("SELECT tags FROM notes WHERE id = ?", (note_id,))
                existing_tags = cursor.fetchone()[0]
                tags_str = existing_tags
                tag_list = existing_tags.split(',') if existing_tags else []

            cursor.execute("""
                UPDATE notes 
                SET content = ?, 
                    updated_at = CURRENT_TIMESTAMP,
                    context = ?,
                    tags = ?
                WHERE id = ?
            """, (content.strip(), json.dumps(new_context), tags_str, note_id))

            # Update tag associations if new tags are provided.
            if tags is not None:
                cursor.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
                for tag in tag_list:
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?", (tag,))
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    cursor.execute("INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)", (note_id, tag_id))

            conn.commit()

            return CommandResult(
                success=True,
                message=f"Updated note {llm_id}\nNew content: {content}\nTags: {', '.join(tag_list)}",
                suggested_commands=[
                    f"notes read {llm_id}",
                    "notes list",
                    f"notes list --tag={tag_list[0]}" if tag_list else "notes tags"
                ],
                data={
                    "note_id": note_id,
                    "llm_id": llm_id,
                    "content": content,
                    "tags": tag_list,
                    "context": new_context
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to update note: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to update note: {str(e)}",
                    suggested_fixes=[
                        "Verify the note ID exists",
                        "Check content format and tags",
                        "Ensure database is accessible"
                    ],
                    related_commands=["notes read", "notes list"],
                    examples=[
                        f'notes update {note_identifier} "Updated content"',
                        f'notes update {note_identifier} "New content" --tags=important,todo'
                    ]
                ),
                suggested_commands=[
                    f"notes read {note_identifier}",
                    "notes list",
                    "notes tags"
                ]
            )
        finally:
            conn.close()
    
    async def _delete_note(self, note_identifier: str) -> CommandResult:
        """
        Delete a note and its associated data.
        
        Cascades removal from:
        - Note content and metadata
        - Tag relationships 
        - Full-text search index via triggers
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Resolve note ID using the helper method
            note_id = self._resolve_note_id(note_identifier, cursor)
            if not note_id:
                return CommandResult(
                    success=False,
                    message=f"Note {note_identifier} not found",
                    suggested_commands=[
                        "notes list",
                        "notes search <query>"
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_identifier} not found",
                        suggested_fixes=[
                            "Check the note ID",
                            "Try 'notes list' to see available notes"
                        ],
                        related_commands=["notes list", "notes search"],
                        examples=["notes delete note-1", "notes list --limit=5"]
                    )
                )

            # Get the llm_id for the response
            cursor.execute("SELECT llm_id FROM notes WHERE id = ?", (note_id,))
            llm_id = cursor.fetchone()[0]

            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
            
            return CommandResult(
                success=True,
                message=f"Successfully deleted note {llm_id}",
                suggested_commands=[
                    "notes list",
                    'notes create "New note"',
                    "notes tags"
                ],
                data={
                    "deleted_id": note_id,
                    "llm_id": llm_id
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to delete note: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to delete note: {str(e)}",
                    suggested_fixes=[
                        "Check the note ID",
                        "Verify database access"
                    ],
                    related_commands=["notes list", "notes create"],
                    examples=[
                        "notes delete note-1",
                        "notes list --limit=5"
                    ]
                ),
                suggested_commands=[
                    "notes list",
                    'notes create "New note"'
                ]
            )
        finally:
            conn.close()
    
    async def _search_notes(self, query: str, limit: int = 5) -> CommandResult:
        """Search notes using full-text search."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT notes.llm_id, notes.content, notes.created_at, notes.tags, notes.sticky
                FROM notes_fts
                JOIN notes ON notes_fts.rowid = notes.rowid
                WHERE notes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, limit))
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResult(
                    success=True,
                    message=f"No notes found matching '{query}'",
                    suggested_commands=[
                        "notes list",
                        f'notes search "{query}" --limit={limit + 5}',
                        f'notes create "Note about {query}"'
                    ],
                    data={
                        "query": query,
                        "count": 0
                    }
                )
            
            content_lines = [f"Search results for '{query}':\n"]
            for llm_id, content, created, tags, sticky in notes:
                sticky_indicator = "📌 " if sticky else ""
                content_lines.append(
                    f"{sticky_indicator}ID: {llm_id}\n"
                    f"Created: {created}\n"
                    f"Content: {content[:50]}...\n"
                    f"Tags: {tags or 'none'}\n"
                )

            return CommandResult(
                success=True,
                message="\n".join(content_lines),
                suggested_commands=[
                    "notes read <note_id>",
                    f'notes search "{query}" --limit={limit + 5}',
                    f'notes create "Note about {query}"'
                ],
                data={
                    "query": query,
                    "count": len(notes),
                    "has_more": len(notes) == limit
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Search failed: {str(e)}",
                error=CommandValidationError(
                    message=f"Search failed: {str(e)}",
                    suggested_fixes=[
                        "Check search query format",
                        "Verify database access"
                    ],
                    related_commands=["notes list", "notes create"],
                    examples=[
                        'notes search "keyword"',
                        'notes search "complex query" --limit=10'
                    ]
                ),
                suggested_commands=[
                    'notes list',
                    'notes create "New note"'
                ]
            )
        finally:
            conn.close()
    
    async def _handle_tags(
        self,
        action: str,
        note_identifier: str,
        *tags: str
    ) -> CommandResult:
        """
        Manage tags for a note.
        Adds or removes tags, and then reassembles the denormalized tags field.
        """
        if action not in ['add', 'remove']:
            return CommandResult(
                success=False,
                message=f"Unknown tag action: {action}",
                suggested_commands=[
                    "notes tag-add <note_id> tag1 tag2",
                    "notes tag-remove <note_id> tag1 tag2"
                ],
                error=CommandValidationError(
                    message=f"Unknown tag action: {action}",
                    suggested_fixes=["Use 'add' or 'remove' only"],
                    related_commands=["notes tag-add", "notes tag-remove"],
                    examples=[
                        "notes tag-add note-1 important",
                        "notes tag-remove note-1 important"
                    ]
                )
            )

        if not tags:
            return CommandResult(
                success=False,
                message="No tags specified",
                suggested_commands=[
                    f"notes tag-{action} {note_identifier} tag1 tag2"
                ],
                error=CommandValidationError(
                    message="No tags specified",
                    suggested_fixes=["Provide at least one tag"],
                    related_commands=["notes tag-add", "notes tag-remove"],
                    examples=[
                        "notes tag-add note-1 important",
                        "notes tag-remove note-1 old-tag"
                    ]
                )
            )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Resolve note ID using the helper method
            note_id = self._resolve_note_id(note_identifier, cursor)
            if not note_id:
                return CommandResult(
                    success=False,
                    message=f"Note {note_identifier} not found",
                    suggested_commands=["notes list"],
                    error=CommandValidationError(
                        message=f"Note {note_identifier} not found",
                        suggested_fixes=[
                            "Check the note ID carefully",
                            "Use 'notes list' to see which IDs exist"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=[
                            "notes tag-add note-1 important",
                            "notes tag-remove note-1 old-tag"
                        ]
                    )
                )

            # Get the llm_id for the response
            cursor.execute("SELECT llm_id FROM notes WHERE id = ?", (note_id,))
            llm_id = cursor.fetchone()[0]

            if action == 'add':
                for tag in tags:
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    cursor.execute("INSERT OR IGNORE INTO note_tags (note_id, tag_id) VALUES (?, ?)", (note_id, tag_id))
            else:  # remove
                for tag in tags:
                    cursor.execute("""
                        DELETE FROM note_tags 
                        WHERE note_id = ? AND tag_id IN (
                            SELECT id FROM tags WHERE name = ?
                        )
                    """, (note_id, tag))
            
            # Reassemble the denormalized tags field.
            cursor.execute("""
                SELECT GROUP_CONCAT(name)
                FROM tags
                WHERE id IN (
                    SELECT tag_id FROM note_tags WHERE note_id = ?
                )
            """, (note_id,))
            new_tags_row = cursor.fetchone()
            new_tags = new_tags_row[0] if new_tags_row and new_tags_row[0] else ""
            
            cursor.execute("UPDATE notes SET tags = ? WHERE id = ?", (new_tags, note_id))
            
            conn.commit()
            
            return CommandResult(
                success=True,
                message=f"{action.capitalize()}ed tags {', '.join(tags)} {'to' if action == 'add' else 'from'} note {llm_id}",
                suggested_commands=[
                    f"notes read {llm_id}",
                    "notes tags",
                    f"notes list --tag={tags[0]}"
                ],
                data={
                    "note_id": note_id,
                    "llm_id": llm_id,
                    "action": action,
                    "tags": list(tags)
                }
            )
        
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to {action} tags: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to {action} tags: {str(e)}",
                    suggested_fixes=[
                        "Check tag names and note ID",
                        "Ensure database access"
                    ],
                    related_commands=["notes list", "notes tags"],
                    examples=[
                        f"notes tag-{action} note-1 important",
                        "notes tags"
                    ]
                ),
                suggested_commands=[
                    "notes list",
                    "notes tags"
                ]
            )
        finally:
            conn.close()

    async def _list_tags(self) -> CommandResult:
        """List all available tags with usage counts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT t.name, COUNT(nt.note_id) as usage_count
                FROM tags t
                LEFT JOIN note_tags nt ON t.id = nt.tag_id
                GROUP BY t.name
                ORDER BY usage_count DESC, t.name
            """)
            
            tags = cursor.fetchall()
            
            if not tags:
                return CommandResult(
                    success=True,
                    message="No tags found",
                    suggested_commands=[
                        'notes create "First note" --tags=example,first',
                        'notes create "Tagged note"'
                    ],
                    data={"count": 0}
                )
            
            content_lines = ["Available tags:\n"]
            for tag, count in tags:
                content_lines.append(f"{tag} ({count} notes)")

            return CommandResult(
                success=True,
                message="\n".join(content_lines),
                suggested_commands=[
                    "notes list --tag=" + tags[0][0],
                    'notes create "New note" --tags=' + tags[0][0],
                    "notes list"
                ],
                data={
                    "tags": dict(tags),
                    "count": len(tags)
                }
            )

        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to list tags: {str(e)}",
                error=CommandValidationError(
                    message=f"Failed to list tags: {str(e)}",
                    suggested_fixes=[
                        "Check database connectivity",
                        "Try again later"
                    ],
                    related_commands=["notes list", "notes create"],
                    examples=["notes tags", "notes list"]
                ),
                suggested_commands=[
                    "notes list",
                    'notes create "New note"'
                ]
            )
        finally:
            conn.close()

    async def _toggle_sticky(self, note_identifier: str) -> CommandResult:
        """Toggle sticky status for a note."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Resolve note ID
            note_id = self._resolve_note_id(note_identifier, cursor)
            if not note_id:
                return CommandResult(
                    success=False,
                    message=f"Note {note_identifier} not found",
                    suggested_commands=["notes list"],
                    error=CommandValidationError(
                        message=f"Note {note_identifier} not found",
                        suggested_fixes=["Check the note ID", "Use 'notes list' to see available notes"],
                        related_commands=["notes list", "notes search"],
                        examples=["notes sticky note-1", "notes sticky note-42"]
                    )
                )
            
            # Get current sticky status and llm_id
            cursor.execute("SELECT sticky, llm_id FROM notes WHERE id = ?", (note_id,))
            current_sticky, llm_id = cursor.fetchone()
            
            new_sticky = not current_sticky
            
            # If making sticky, check limit
            if new_sticky:
                cursor.execute("SELECT COUNT(*) FROM notes WHERE sticky = 1")
                current_sticky_count = cursor.fetchone()[0]
                
                if current_sticky_count >= self.max_sticky_notes:
                    return CommandResult(
                        success=False,
                        message=f"Cannot make note sticky: already have {current_sticky_count}/{self.max_sticky_notes} sticky notes",
                        suggested_commands=["notes list", f"notes sticky <note_id>  # to unstick another"],
                        error=CommandValidationError(
                            message=f"Sticky limit ({self.max_sticky_notes}) exceeded",
                            suggested_fixes=[
                                "Remove sticky from another note first",
                                "Check current sticky notes"
                            ],
                            related_commands=["notes list"],
                            examples=["notes sticky note-1  # to unstick", "notes list"]
                        )
                    )
            
            # Update sticky status
            cursor.execute("UPDATE notes SET sticky = ? WHERE id = ?", (new_sticky, note_id))
            conn.commit()
            
            status = "sticky" if new_sticky else "unsticky"
            return CommandResult(
                success=True,
                message=f"Note {llm_id} is now {status}",
                suggested_commands=[
                    f"notes read {llm_id}",
                    "notes list"
                ],
                data={
                    "note_id": note_id,
                    "llm_id": llm_id,
                    "sticky": new_sticky
                }
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Failed to toggle sticky: {str(e)}",
                suggested_commands=["notes list"]
            )
        finally:
            conn.close()

    def _get_next_llm_id(self, cursor) -> str:
        """Get the next available LLM ID."""
        cursor.execute("SELECT MAX(CAST(SUBSTR(llm_id, 6) AS INTEGER)) FROM notes WHERE llm_id LIKE 'note-%'")
        result = cursor.fetchone()
        next_num = (result[0] or 0) + 1
        return f"note-{next_num}"

    def _resolve_note_id(self, note_identifier: str, cursor) -> Optional[str]:
        """Resolve either UUID or llm_id to the actual UUID primary key."""
        note_identifier = note_identifier.strip()
        
        # Try as UUID first (longer strings)
        if len(note_identifier) > 10:  # UUIDs are much longer than our llm IDs
            cursor.execute("SELECT id FROM notes WHERE id = ?", (note_identifier,))
            result = cursor.fetchone()
            if result:
                return result[0]

        # Try as llm_id
        cursor.execute("SELECT id FROM notes WHERE llm_id = ?", (note_identifier,))
        result = cursor.fetchone()
        if result:
            return result[0]
            
        return None
