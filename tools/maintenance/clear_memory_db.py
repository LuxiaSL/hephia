"""
full_db_cleanup_standalone.py

A standalone utility script to perform a complete and total cleanup of the memory database.
This script will:
1. Drop all existing tables.
2. Vacuum the database to reclaim disk space and reduce file size.
3. Recreate the database schema from its internal definition.

This script is self-contained and does not require any external project files.
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List

# --- Start of inlined schema.py content ---

# Common field sizes
MAX_TEXT_LENGTH = 65535
DEFAULT_STRING_LENGTH = 255

# Node states
NODE_STATES = {
    'ACTIVE': 'active',
    'GHOSTED': 'ghosted',
    'PRUNED': 'pruned'
}

# Relationship types
MEMORY_LINK_TYPES = {
    'DIRECT': 'direct',
    'TEMPORAL': 'temporal',
    'MERGED': 'merged',
    'RESURRECT': 'resurrection'
}

SYNTHESIS_TYPES = {
    'CONFLICT': 'conflict_synthesis',
    'MERGE': 'merge_synthesis',
    'RESURRECTION': 'resurrection'
}

# Body Memory Tables
BODY_MEMORY_SCHEMA = {
    'body_memory_nodes': """
        CREATE TABLE IF NOT EXISTS body_memory_nodes (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            raw_state TEXT NOT NULL,
            processed_state TEXT NOT NULL,
            strength REAL NOT NULL,
            ghosted BOOLEAN DEFAULT FALSE,
            parent_node_id INTEGER,
            ghost_nodes TEXT,
            ghost_states TEXT,
            connections TEXT,
            last_connection_update REAL,
            last_accessed REAL,
            FOREIGN KEY(parent_node_id) REFERENCES body_memory_nodes(id)
        )
    """,
    'body_memory_indexes': [
        "CREATE INDEX IF NOT EXISTS idx_body_timestamp ON body_memory_nodes(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_body_strength ON body_memory_nodes(strength)",
        "CREATE INDEX IF NOT EXISTS idx_body_ghosted ON body_memory_nodes(ghosted)",
        "CREATE INDEX IF NOT EXISTS idx_body_last_accessed ON body_memory_nodes(last_accessed)"
    ]
}

# Cognitive Memory Tables
COGNITIVE_MEMORY_SCHEMA = {
    'cognitive_memory_nodes': """
        CREATE TABLE IF NOT EXISTS cognitive_memory_nodes (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            text_content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            raw_state TEXT NOT NULL,
            processed_state TEXT NOT NULL,
            strength REAL NOT NULL,
            ghosted BOOLEAN DEFAULT FALSE,
            parent_node_id INTEGER,
            ghost_nodes TEXT,
            ghost_states TEXT,
            connections TEXT,
            semantic_context TEXT,
            last_accessed REAL,
            formation_source TEXT,
            last_echo_time REAL,
            echo_dampening REAL DEFAULT 1.0,
            last_connection_update REAL,
            FOREIGN KEY(parent_node_id) REFERENCES cognitive_memory_nodes(id)
        )
    """,
    'cognitive_memory_indexes': [
        "CREATE INDEX IF NOT EXISTS idx_cognitive_timestamp ON cognitive_memory_nodes(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_strength ON cognitive_memory_nodes(strength)",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_ghosted ON cognitive_memory_nodes(ghosted)",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_last_accessed ON cognitive_memory_nodes(last_accessed)",
        "CREATE INDEX IF NOT EXISTS idx_cognitive_last_connection_update ON cognitive_memory_nodes(last_connection_update)"
    ]
}

# Relationship Tables
RELATIONSHIP_SCHEMA = {
    'memory_links': """
        CREATE TABLE IF NOT EXISTS memory_links (
            id INTEGER PRIMARY KEY,
            cognitive_node_id INTEGER NOT NULL,
            body_node_id INTEGER NOT NULL,
            link_strength REAL NOT NULL,
            link_type TEXT NOT NULL,
            created_at REAL NOT NULL,
            metadata TEXT,
            UNIQUE(cognitive_node_id, body_node_id),
            FOREIGN KEY(cognitive_node_id) REFERENCES cognitive_memory_nodes(id),
            FOREIGN KEY(body_node_id) REFERENCES body_memory_nodes(id)
        )
    """,
    'synthesis_relations': """
        CREATE TABLE IF NOT EXISTS synthesis_relations (
            id INTEGER PRIMARY KEY,
            synthesis_node_id INTEGER NOT NULL,
            constituent_node_id INTEGER NOT NULL,
            relationship_type TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY(synthesis_node_id) REFERENCES cognitive_memory_nodes(id),
            FOREIGN KEY(constituent_node_id) REFERENCES cognitive_memory_nodes(id)
        )
    """,
    'relationship_indexes': [
        "CREATE INDEX IF NOT EXISTS idx_memory_links_cognitive ON memory_links(cognitive_node_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_links_body ON memory_links(body_node_id)",
        "CREATE INDEX IF NOT EXISTS idx_synthesis_synthesis ON synthesis_relations(synthesis_node_id)",
        "CREATE INDEX IF NOT EXISTS idx_synthesis_constituent ON synthesis_relations(constituent_node_id)"
    ]
}

# Table Creation Functions
COMPLETE_MEMORY_SCHEMA = {
    **BODY_MEMORY_SCHEMA,
    **COGNITIVE_MEMORY_SCHEMA,
    **RELATIONSHIP_SCHEMA
}

TABLE_CREATION_ORDER = [
    'body_memory_nodes',
    'cognitive_memory_nodes',
    'memory_links',
    'synthesis_relations'
]

def get_complete_schema() -> Dict[str, str]:
    """Returns complete schema including all tables and indexes."""
    return COMPLETE_MEMORY_SCHEMA

def get_table_creation_order() -> List[str]:
    """Returns correct order for table creation."""
    return TABLE_CREATION_ORDER

def get_all_indexes() -> List[str]:
    """Returns all index creation statements."""
    return (
        BODY_MEMORY_SCHEMA['body_memory_indexes'] +
        COGNITIVE_MEMORY_SCHEMA['cognitive_memory_indexes'] +
        RELATIONSHIP_SCHEMA['relationship_indexes']
    )

# --- End of inlined schema.py content ---

# Default database path
DEFAULT_DB_PATH = "data/memory.db"

def full_db_cleanup(db_path: str = DEFAULT_DB_PATH) -> None:
    """
    Completely drops all tables, vacuums the database, and recreates the schema.

    Args:
        db_path: Path to the SQLite database file.
    """
    print(f"Starting full cleanup for database: {db_path}")

    try:
        # Ensure the parent directory for the database exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- 1. Drop all existing tables ---
        print("--- Step 1: Dropping all tables ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if tables:
            cursor.execute("PRAGMA foreign_keys = OFF")
            for (table_name,) in tables:
                if table_name != "sqlite_sequence":  # Skip internal SQLite table
                    print(f"Dropping table: {table_name}")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute("PRAGMA foreign_keys = ON")
            print("All tables dropped successfully.")
        else:
            print("No tables found to drop.")

        # --- 2. Vacuum the database ---
        print("\n--- Step 2: Vacuuming the database ---")
        try:
            # Setting isolation_level to None commits the VACUUM immediately.
            conn.close()
            conn = sqlite3.connect(db_path, isolation_level=None)
            conn.execute("VACUUM")
            conn.close()
            print("Database vacuumed successfully, file size has been reduced.")
        except Exception as e:
            print(f"Could not complete VACUUM: {e}")
            # Reconnect for the next steps
            conn = sqlite3.connect(db_path)

        # --- 3. Recreate the schema ---
        print("\n--- Step 3: Recreating database schema ---")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get schema and creation order from inlined functions
        table_creation_order = get_table_creation_order()
        schema = get_complete_schema()

        # Create tables in the correct order
        for table_name in table_creation_order:
            if 'indexes' not in table_name:  # Avoid trying to execute the index list as a table
                print(f"Creating table: {table_name}")
                cursor.execute(schema[table_name])

        # Create all indexes
        print("Creating indexes...")
        all_indexes = get_all_indexes()
        for index_statement in all_indexes:
            cursor.execute(index_statement)
        
        print("Schema and indexes recreated successfully.")

        # Commit all changes and close the connection
        conn.commit()
        conn.close()

        print(f"\nDatabase cleanup and reset complete for: {db_path}")

    except Exception as e:
        print(f"An error occurred during the cleanup process: {e}")
        raise

def main():
    """Main entry point"""
    db_path = os.getenv("MEMORY_DB_PATH", DEFAULT_DB_PATH)
    
    print("=========================================")
    print("  STANDALONE DATABASE CLEANUP UTILITY  ")
    print("=========================================")
    print(f"Target Database: {db_path}\n")

    # A simple confirmation prompt to prevent accidental data loss
    try:
        confirm = input(
            "WARNING: This will permanently delete all data. "
            "Are you sure you want to continue? (yes/no): "
        )
        if confirm.lower() != 'yes':
            print("Operation cancelled.")
            return
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return

    full_db_cleanup(db_path)
    print("\nCleanup operation finished.")

if __name__ == "__main__":
    main()