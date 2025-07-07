"""
memory/db/schema.py

Defines the complete database schema for the memory system.
Includes table definitions for both body and cognitive memory systems
along with their relationships and shared constants.
"""

from typing import Dict, List

# -----------------------------------------------------------------------------
# 1. Schema Constants
# -----------------------------------------------------------------------------

# Common field sizes
MAX_TEXT_LENGTH = 65535  # For larger text fields
DEFAULT_STRING_LENGTH = 255

# Node states
NODE_STATES = {
    'ACTIVE': 'active',
    'GHOSTED': 'ghosted',
    'PRUNED': 'pruned'
}

# Relationship types
MEMORY_LINK_TYPES = {
    'DIRECT': 'direct',           # Direct formation link
    'TEMPORAL': 'temporal',       # Temporally related
    'MERGED': 'merged',          # Result of merge
    'RESURRECT': 'resurrection'   # From resurrection
}

SYNTHESIS_TYPES = {
    'CONFLICT': 'conflict_synthesis',
    'MERGE': 'merge_synthesis',
    'RESURRECTION': 'resurrection'
}

# -----------------------------------------------------------------------------
# 2. Body Memory Tables
# -----------------------------------------------------------------------------

BODY_MEMORY_SCHEMA = {
    'body_memory_nodes': """
        CREATE TABLE IF NOT EXISTS body_memory_nodes (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            raw_state TEXT NOT NULL,        -- Emotional vectors, needs, etc.
            processed_state TEXT NOT NULL,   -- Processed/summarized states
            strength REAL NOT NULL,
            ghosted BOOLEAN DEFAULT FALSE,
            parent_node_id INTEGER,
            ghost_nodes TEXT,               -- Serialized array of ghost nodes
            ghost_states TEXT,              -- Serialized array of past states
            connections TEXT,               -- Serialized connection mapping
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

# -----------------------------------------------------------------------------
# 3. Cognitive Memory Tables
# -----------------------------------------------------------------------------

COGNITIVE_MEMORY_SCHEMA = {
    'cognitive_memory_nodes': """
        CREATE TABLE IF NOT EXISTS cognitive_memory_nodes (
            id INTEGER PRIMARY KEY,
            timestamp REAL NOT NULL,
            text_content TEXT NOT NULL,
            embedding TEXT NOT NULL,         -- Serialized embedding vector
            raw_state TEXT NOT NULL,
            processed_state TEXT NOT NULL,
            strength REAL NOT NULL,
            ghosted BOOLEAN DEFAULT FALSE,
            parent_node_id INTEGER,
            ghost_nodes TEXT,               -- Serialized ghost node array
            ghost_states TEXT,              -- Serialized state history
            connections TEXT,               -- Serialized connection mapping
            semantic_context TEXT,          -- Additional semantic metadata
            last_accessed REAL,
            formation_source TEXT,          -- Event/trigger that created memory
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

# -----------------------------------------------------------------------------
# 4. Relationship Tables
# -----------------------------------------------------------------------------

RELATIONSHIP_SCHEMA = {
    'memory_links': """
        CREATE TABLE IF NOT EXISTS memory_links (
            id INTEGER PRIMARY KEY,
            cognitive_node_id INTEGER NOT NULL,
            body_node_id INTEGER NOT NULL,
            link_strength REAL NOT NULL,
            link_type TEXT NOT NULL,
            created_at REAL NOT NULL,
            metadata TEXT,                   -- State signatures and other metadata
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
            metadata TEXT,                   -- Synthesis details and metrics
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

# -----------------------------------------------------------------------------
# 5. Table Creation Functions
# -----------------------------------------------------------------------------

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