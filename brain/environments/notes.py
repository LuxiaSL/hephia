"""
notes.py - Persistent note-taking environment for Hephia.

Provides a structured note management system with tagging and search capabilities.
Maintains SQLite database for storage while exposing a clean command interface.
"""

import sqlite3
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from uuid import uuid4

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
        self._ensure_db()  # Setup database first
        super().__init__()  # Then initialize command structure
        
        self.help_text = """
        The notes environment provides persistent storage for thoughts and observations.
        
        Core Features:
        - Create and manage notes with content and tags
        - Search notes by content or tags
        - Track note context and metadata
        - Organize with tagging system
        
        Tips:
        - Use tags consistently for better organization
        - Include context in searches
        - Review and update notes regularly
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
                    )
                ],
                examples=[
                    'notes create "<Content of note>"',
                    'notes create "<Content of note>" --tags=tags,go,here',
                ],
                related_commands=['list', 'tags'],
                failure_hints={
                    "DatabaseError": "Failed to save note. Try again.",
                    "ValueError": "Invalid note format. Use quotes around content."
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
                        required=True
                    )
                ],
                examples=[
                    'notes read abc123',
                    'notes read def456'
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
                        required=True
                    ),
                    Parameter(
                        name="content",
                        description="The new note content",
                        required=True,
                        examples=['"Updated content"']
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
                    'notes update abc123 "New content"',
                ],
                related_commands=['read', 'delete'],
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
                        required=True
                    )
                ],
                examples=[
                    'notes delete abc123'
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
            description="List all available tags", 
            parameters=[],  # No parameters needed
            examples=[
                'notes tags'  # Simply list tags
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
                required=True
                ),
                Parameter(
                name="tags",
                description="Space-separated tags to add",
                required=True,
                examples=["important", "todo project"]
                )
            ],
            examples=[
                'notes tag-add abc123 important',
                'notes tag-add def456 project idea'
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
                required=True
                ),
                Parameter(
                name="tags", 
                description="Space-separated tags to remove",
                required=True,
                examples=["important", "todo project"]
                )
            ],
            examples=[
                'notes tag-remove abc123 important',
                'notes tag-remove def456 old-tag'
            ],
            failure_hints={
                "NotFound": "Note not found. Check the ID.",
                "ValueError": "Invalid tags format. Use space-separated tags.",
                "DatabaseError": "Failed to remove tags. Try again." 
            },
            related_commands=['tags', 'tag-add']
            )
        )
    
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
                return await self._create_note(params[0], context, flags.get("tags"))
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
            
            raise ValueError(f"Unknown action: {action}")
            
        except sqlite3.Error as e:
            return CommandResult(
                success=False,
                message=f"Database error: {str(e)}",
                error=CommandValidationError(
                    message=f"Database error: {str(e)}",
                    suggested_fixes=[
                        "Check database connectivity and ensure no concurrency issues",
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
    ) -> CommandResult:
        """
        Create a new note with context awareness.
        
        The note creation ripples through the system:
        - Stores the current cognitive context
        - Updates tag relationships
        - Suggests natural next actions
        """
        note_id = str(uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Capture current cognitive state
            note_context = {
                'mood': context.get('internal_state', {}).get('mood', {}),
                'needs': context.get('internal_state', {}).get('needs', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store the note
            cursor.execute(
                """INSERT INTO notes (id, content, context) 
                   VALUES (?, ?, ?)""",
                (note_id, content.strip(), json.dumps(note_context))
            )
            
            # Handle tags if provided
            tag_list = []
            if tags:
                tag_list = [t.strip() for t in tags.split(',')]
                for tag in tag_list:
                    # Insert or get tag
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                        (tag,)
                    )
                    cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    
                    # Link to note
                    cursor.execute(
                        "INSERT INTO note_tags (note_id, tag_id) VALUES (?, ?)",
                        (note_id, tag_id)
                    )
            
            conn.commit()
            
            # Notify system
            global_event_dispatcher.dispatch_event(
                Event("notes:created", {
                    "note_id": note_id,
                    "content": content,
                    "tags": tag_list,
                    "context": note_context
                })
            )
            
            # Generate contextual next steps
            suggested_next = [
                f"notes read {note_id}",  # View the note
                "notes list"  # See in context
            ]
            
            # Add tag-based suggestions
            if tags:
                suggested_next.append(f"notes list --tag={tag_list[0]}")  # View similar notes
            else:
                suggested_next.append(f'notes tag-add {note_id} <tag>')  # Suggest tagging
            
            return CommandResult(
                success=True,
                message=f"Created note {note_id}\nContent: {content}\nTags: {', '.join(tag_list) if tag_list else 'none'}",
                suggested_commands=suggested_next,
                data={
                    "note_id": note_id,
                    "content": content,
                    "tags": tag_list,
                    "context": note_context
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

    async def _read_note(self, note_id: str) -> CommandResult:
        """
        Read a specific note with its full context.
        
        Reading flows naturally into:
        - Updating the note
        - Exploring related notes
        - Managing tags
        """
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
                return CommandResult(
                    success=False,
                    message=f"Note {note_id} not found",
                    suggested_commands=[
                        "notes list",
                        'notes create "New note"'
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_id} not found",
                        suggested_fixes=[
                            "Check the note ID",
                            "Try 'notes list' to see available notes"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=["notes read abc123", "notes list --limit=5"]
                    )
                )
            
            # Parse stored context
            context = json.loads(note[4]) if note[4] else {}
            
            # Format note display
            content = [
                f"Note {note[0]}:",
                f"Created: {note[2]}",
                f"Updated: {note[3]}",
                f"Tags: {note[-1] or 'none'}",
                "",
                "Content:",
                note[1]
            ]
            
            if context.get('mood'):
                content.extend([
                    "",
                    "Created while:",
                    f"Mood: {context['mood'].get('name', 'unknown')}",
                    f"Needs: {', '.join(f'{k}: {v}' for k, v in context.get('needs', {}).items())}"
                ])

            # Generate contextual suggestions
            suggested = [
                f"notes update {note_id} \"New content\"",  # Update
                f"notes tag-add {note_id} <tag>"  # Add tags
            ]
            
            # Add tag-based suggestions
            if note[-1]:  # Has tags
                tags = note[-1].split(',')
                suggested.append(f"notes list --tag={tags[0]}")  # View similar notes
            
            return CommandResult(
                success=True,
                message="\n".join(content),
                suggested_commands=suggested,
                data={
                    "note_id": note_id,
                    "content": note[1],
                    "created": note[2],
                    "updated": note[3],
                    "tags": note[-1].split(',') if note[-1] else [],
                    "context": context
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
                        "notes read abc123",
                        "notes list --limit=5",
                        'notes search "keyword"'
                    ]
                ),
                suggested_commands=[
                    "notes list", 
                    "notes search <query>",
                    'notes create "New note"'
                ]
            )
            
        finally:
            conn.close()

    
    async def _list_notes(self, tag: Optional[str] = None, limit: int = 10) -> CommandResult:
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
            if tag:
                query += " WHERE t.name = ?"
                cursor.execute(query + f" GROUP BY n.id ORDER BY n.created_at DESC LIMIT {limit}", (tag,))
            else:
                cursor.execute(query + f" GROUP BY n.id ORDER BY n.created_at DESC LIMIT {limit}")
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResult(
                success=True,  # Still successful, just empty!
                message="No notes found.",
                suggested_commands=[
                    'notes create "My first note" --tags=example',
                    'notes create "Getting started"'
                ],
                data={"count": 0}  # Include metadata
            )
            
            # Format notes list
            content = ["Recent Notes:\n"]
            for note in notes:
                content.append(
                    f"ID: {note[0]}\n"
                    f"Created: {note[2]}\n"
                    f"Content: {note[1][:75]}...\n"
                    f"Tags: {note[-1] or 'none'}\n"
                )

            return CommandResult(
                success=True,
                message="\n".join(content),
                suggested_commands=[
                    "notes read <note_id>",
                    "notes search <query>",
                    f"notes list --limit={limit + 5}"  # Suggest viewing more
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
                    'notes create "New note"',
                    "notes tags",
                    "notes search <query>"
                ]
            )
            
        finally:
            conn.close()
    
    async def _update_note(
        self,
        note_id: str,
        content: str,
        context: Dict[str, Any],
        tags: Optional[str] = None,
    ) -> CommandResult:
        """Update a note's content and metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First verify note exists
            cursor.execute("""
                SELECT EXISTS(SELECT 1 FROM notes WHERE id = ?)
            """, (note_id,))
            
            if not cursor.fetchone()[0]:
                return CommandResult(
                    success=False,
                    message=f"Note {note_id} not found",
                    suggested_commands=[
                        "notes list",
                        f'notes create "{content}"'
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_id} not found",
                        suggested_fixes=[
                            "Verify the note ID is correct",
                            "Use 'notes list' to see available note IDs"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=[f'notes update {note_id} "New content"', "notes list --limit=5"]
                    )
                )

            # Update note content and context
            new_context = {
                'mood': context.get('internal_state', {}).get('mood', {}),
                'needs': context.get('internal_state', {}).get('needs', {}),
                'timestamp': datetime.now().isoformat()
            }
            
            cursor.execute("""
                UPDATE notes 
                SET content = ?, 
                    updated_at = CURRENT_TIMESTAMP,
                    context = ?
                WHERE id = ?
            """, (content.strip(), json.dumps(new_context), note_id))

            # Handle tags if provided
            if tags is not None:
                # Remove existing tags
                cursor.execute("DELETE FROM note_tags WHERE note_id = ?", (note_id,))
                
                # Add new tags
                tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                for tag in tag_list:
                    # Insert or get tag
                    cursor.execute(
                        "INSERT OR IGNORE INTO tags (name) VALUES (?)",
                        (tag,)
                    )
                    cursor.execute(
                        "SELECT id FROM tags WHERE name = ?",
                        (tag,)
                    )
                    tag_id = cursor.fetchone()[0]
                    
                    # Link tag to note
                    cursor.execute("""
                        INSERT INTO note_tags (note_id, tag_id) 
                        VALUES (?, ?)
                    """, (note_id, tag_id))

            conn.commit()

            # Get updated tags using LEFT JOIN structure
            cursor.execute("""
                SELECT GROUP_CONCAT(t.name)
                FROM notes n
                LEFT JOIN note_tags nt ON n.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
                WHERE n.id = ?
                GROUP BY n.id
            """, (note_id,))
            
            current_tags = cursor.fetchone()
            tag_list = current_tags[0].split(',') if current_tags and current_tags[0] else []

            return CommandResult(
                success=True,
                message=f"Updated note {note_id}\nNew content: {content}\nTags: {', '.join(tag_list)}",
                suggested_commands=[
                    f"notes read {note_id}",
                    "notes list",
                    f"notes list --tag={tag_list[0]}" if tag_list else "notes tags"
                ],
                data={
                    "note_id": note_id,
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
                        f'notes update {note_id} "Updated content"',
                        f'notes update {note_id} "New content" --tags=important,todo'
                    ]
                ),
                suggested_commands=[
                    f"notes read {note_id}",
                    "notes list",
                    "notes tags"
                ]
            )

        finally:
            conn.close()
    
    async def _delete_note(self, note_id: str) -> CommandResult:
        """
        Delete a note and its associated data.
        
        Deletion ripples through:
        - Note content and metadata
        - Tag relationships 
        - Search index
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First check if note exists
            cursor.execute("SELECT 1 FROM notes WHERE id = ?", (note_id.strip(),))
            if not cursor.fetchone():
                return CommandResult(
                    success=False,
                    message=f"Note {note_id} not found",
                    suggested_commands=[
                        "notes list",
                        "notes search <query>"
                    ],
                    error=CommandValidationError(
                        message=f"Note {note_id} not found",
                        suggested_fixes=[
                            "Double-check the note ID",
                            "Try 'notes list' to see existing notes"
                        ],
                        related_commands=["notes list", "notes search"],
                        examples=["notes delete abc123", "notes list --limit=5"]
                    )
                )

            # Delete note - tags are handled by foreign key constraints
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id.strip(),))
            conn.commit()
            
            return CommandResult(
                success=True,
                message=f"Successfully deleted note {note_id}",
                suggested_commands=[
                    "notes list",  # View remaining notes
                    "notes create \"New note\"",  # Create new note
                    "notes tags"  # Review available tags
                ],
                data={
                    "deleted_id": note_id
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
                    examples=["notes delete abc123", "notes list --limit=5"]
                ),
                suggested_commands=[
                    "notes list",
                    "notes create \"New note\""
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
                SELECT notes.*, GROUP_CONCAT(t.name) as tags
                FROM notes_fts
                JOIN notes ON notes_fts.rowid = notes.rowid
                LEFT JOIN note_tags nt ON notes.id = nt.note_id
                LEFT JOIN tags t ON nt.tag_id = t.id
                WHERE notes_fts MATCH ?
                GROUP BY notes.id
                ORDER BY rank
                LIMIT 5
            """, (query, limit))
            
            notes = cursor.fetchall()
            
            if not notes:
                return CommandResult(
                    success=True,  # Still a successful search, just no results
                    message=f"No notes found matching '{query}'",
                    suggested_commands=[
                        "notes list",  # View all notes instead
                        f'notes search "{query}" --limit={limit + 5}',  # Try more results
                        f'notes create "New note about {query}"'  # Create new note
                    ],
                    data={
                        "query": query,
                        "count": 0
                    }
                )
            
            # Format search results
            content = [f"Search results for '{query}':\n"]
            for note in notes:
                content.append(
                    f"ID: {note[0]}\n"
                    f"Content: {note[1][:50]}...\n"
                    f"Tags: {note[-1] or 'none'}\n"
                )

            return CommandResult(
                success=True,
                message="\n".join(content),
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
        note_id: str,
        *tags: str
    ) -> CommandResult:
        """
        Manage tags for a note.
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
                        "notes tag-add abc123 important",
                        "notes tag-remove abc123 important"
                    ]
                )
            )

        if not tags:
            return CommandResult(
                success=False,
                message="No tags specified",
                suggested_commands=[
                    f"notes tag-{action} {note_id} tag1 tag2"
                ],
                error=CommandValidationError(
                    message="No tags specified",
                    suggested_fixes=["Provide at least one tag"],
                    related_commands=["notes tag-add", "notes tag-remove"],
                    examples=[
                        "notes tag-add abc123 important",
                        "notes tag-remove abc123 old-tag"
                    ]
                )
            )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Verify note exists
            cursor.execute("SELECT 1 FROM notes WHERE id = ?", (note_id,))
            if not cursor.fetchone():
                return CommandResult(
                    success=False,
                    message=f"Note {note_id} not found",
                    suggested_commands=["notes list"],
                    error=CommandValidationError(
                        message=f"Note {note_id} not found",
                        suggested_fixes=[
                            "Check the note ID carefully",
                            "Use 'notes list' to see which IDs exist"
                        ],
                        related_commands=["notes list", "notes create"],
                        examples=[
                            "notes tag-add abc123 important",
                            "notes tag-remove abc123 old-tag"
                        ]
                    )
                )


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
            
            return CommandResult(
                success=True,
                message=f"{action.capitalize()}ed tags {', '.join(tags)} {'to' if action == 'add' else 'from'} note {note_id}",
                suggested_commands=[
                    f"notes read {note_id}",  # View updated note
                    "notes tags",  # View all tags
                    f"notes list --tag={tags[0]}"  # View notes with this tag
                ],
                data={
                    "note_id": note_id,
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
                        f"notes tag-{action} abc123 important",
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
            
            # Format tag list with counts
            content = ["Available tags:\n"]
            for tag, count in tags:
                content.append(f"{tag} ({count} notes)")

            return CommandResult(
                success=True,
                message="\n".join(content),
                suggested_commands=[
                    "notes list --tag=" + tags[0][0],  # List notes with most used tag
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