#internal\modules\memory\db\operations.py

import aiosqlite
import asyncio
import json
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass

from loggers.loggers import MemoryLogger

from .schema import (
    get_complete_schema,
    get_all_indexes,
    get_table_creation_order
)
from ..state.signatures import BodyStateSignature


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class BaseDBOperations:
    """
    Handles core database operations for both body and cognitive memory systems.
    Manages connections, transactions, CRUD for main tables,
    cross-memory references, and synthesis relations.
    """

    def __init__(self, db_path: str = 'data/memory.db'):
        self.db_path = db_path
        self.logger = MemoryLogger

    # -------------------------------------------------------------------------
    # 1) CONNECTION & TRANSACTION
    # -------------------------------------------------------------------------
    @asynccontextmanager
    async def timeout_context(self, seconds: float = 30.0):
        try:
            async with asyncio.timeout(seconds):
                yield
        except asyncio.TimeoutError:
            raise DatabaseError("Operation timed out")
        
    @asynccontextmanager
    async def _db_connection(self, read_only: bool = False, timeout: float = 30.0):
        async with self.timeout_context(timeout):
            conn = await aiosqlite.connect(self.db_path)
            if read_only:
                await conn.execute("BEGIN DEFERRED")
            try:
                yield conn
                if not read_only:
                    await conn.commit()
            except Exception as e:
                await conn.rollback()
                self.logger.error(f"Database connection failed: {str(e)}")
                raise
            finally:
                await conn.close()

    @asynccontextmanager
    async def transaction(self):
        async with self._db_connection() as conn:
            try:
                yield conn
            except Exception as e:
                await conn.rollback()
                self.logger.error(f"Transaction rolled back: {str(e)}")
                raise

    # -------------------------------------------------------------------------
    # 2) SCHEMA CREATION & MAINTENANCE
    # -------------------------------------------------------------------------
    async def init_database(self) -> None:
        """Initialize the entire database schema (tables + indexes)."""
        async with self.transaction() as conn:
            async with conn.cursor() as cursor:
                # Create tables in the correct order
                for table_name in get_table_creation_order():
                    table_sql = get_complete_schema()[table_name]
                    await cursor.execute(table_sql)

                # Create indexes
                for index_sql in get_all_indexes():
                    await cursor.execute(index_sql)

            self.logger.debug("Database schema initialized successfully.")

    async def vacuum_database(self) -> None:
        """Optimize database storage."""
        async with self.transaction() as conn:
            await conn.execute("VACUUM")
        self.logger.debug("Database vacuum completed.")

    async def integrity_check(self) -> bool:
        """Perform a database integrity verification."""
        result = await self._execute_query("PRAGMA integrity_check")
        valid = (len(result) == 1 and result[0][0] == "ok")
        self.logger.debug(f"Integrity check result: {valid}")
        return valid

    async def backup_database(self, backup_path: str) -> None:
        """Create a full database backup (copies the entire DB to backup_path)."""
        async with self._db_connection(read_only=True) as src, \
                   aiosqlite.connect(backup_path) as dst:
            await src.backup(dst)
        self.logger.debug(f"Database backup created at: {backup_path}")

    # -------------------------------------------------------------------------
    # 3) GENERIC QUERY EXECUTOR
    # -------------------------------------------------------------------------
    async def _execute_query(
        self,
        query: str,
        params: tuple = (),
        conn: Optional[aiosqlite.Connection] = None
    ) -> List[Any]:
        """
        Generic query executor with optional external connection.
        Automatically decides read-only vs. read-write context.
        """
        try:
            if conn is None:
                is_select = query.strip().upper().startswith("SELECT")
                async with self._db_connection(read_only=is_select) as c:
                    return await self._execute_query(query, params, conn=c)

            async with conn.cursor() as cursor:
                await cursor.execute(query, params)
                if cursor.description is None:  # Non-SELECT queries
                    return []
                rows = await cursor.fetchall()
                return rows

        except Exception as e:
            self.logger.error(f"Query failed: {query} | {params} | {str(e)}")
            raise DatabaseError(f"Query execution error: {str(e)}")

    # -------------------------------------------------------------------------
    # 4) BODY MEMORY CRUD
    # -------------------------------------------------------------------------
    async def create_body_node(self, node_data: Dict[str, Any]) -> int:
        """
        Inserts a new record into 'body_memory_nodes'.
        Returns the newly created node's ID.
        """
        query = """
            INSERT INTO body_memory_nodes (
                timestamp, raw_state, processed_state, strength, ghosted,
                parent_node_id, ghost_nodes, ghost_states, connections,
                last_connection_update
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            node_data['timestamp'],
            node_data['raw_state'],
            node_data['processed_state'],
            node_data['strength'],
            node_data.get('ghosted', False),
            node_data.get('parent_node_id'),
            node_data.get('ghost_nodes', '[]'),
            node_data.get('ghost_states', '[]'),
            node_data.get('connections', '{}'),
            node_data.get('last_connection_update')
        )
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT last_insert_rowid()")
                row = await cursor.fetchone()
                new_id = row[0]
        self.logger.debug(f"Created new body_memory_node with ID {new_id}")
        return new_id

    async def get_body_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single body_memory_node record by ID."""
        query = "SELECT * FROM body_memory_nodes WHERE id = ?"
        results = await self._execute_query(query, (node_id,))
        if not results:
            self.logger.warning(f"No body_memory_node found with ID {node_id}")
            return None

        columns = [
            'id', 'timestamp', 'raw_state', 'processed_state', 'strength',
            'ghosted', 'parent_node_id', 'ghost_nodes', 'ghost_states',
            'connections', 'last_connection_update'
        ]
        row = results[0]
        node_dict = dict(zip(columns, row))
        self.logger.debug(f"Retrieved body_memory_node: {node_dict}")
        return node_dict

    async def update_body_node(self, node_id: int, node_data: Dict[str, Any]) -> None:
        """Updates an existing body_memory_node by ID."""
        query = """
            UPDATE body_memory_nodes
            SET timestamp = ?, raw_state = ?, processed_state = ?, strength = ?,
                ghosted = ?, parent_node_id = ?, ghost_nodes = ?, ghost_states = ?,
                connections = ?, last_connection_update = ?
            WHERE id = ?
        """
        params = (
            node_data['timestamp'],
            node_data['raw_state'],
            node_data['processed_state'],
            node_data['strength'],
            node_data.get('ghosted', False),
            node_data.get('parent_node_id'),
            node_data.get('ghost_nodes', '[]'),
            node_data.get('ghost_states', '[]'),
            node_data.get('connections', '{}'),
            node_data.get('last_connection_update'),
            node_id
        )
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
            if conn.total_changes == 0:
                raise DatabaseError(f"Body node {node_id} not found for update.")
        self.logger.debug(f"Updated body_memory_node {node_id}")

    async def delete_body_node(self, node_id: int) -> None:
        """Removes a body_memory_node by ID (and potential referencing data if needed)."""
        query = "DELETE FROM body_memory_nodes WHERE id = ?"
        async with self.transaction() as conn:
            await self._execute_query(query, (node_id,), conn=conn)
            if conn.total_changes == 0:
                raise DatabaseError(f"BodyMemoryNode {node_id} not found for deletion.")
        self.logger.debug(f"Deleted body_memory_node {node_id}")

    async def load_all_body_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all body_memory_nodes from the DB."""
        query = "SELECT * FROM body_memory_nodes"
        results = await self._execute_query(query)
        columns = [
            'id', 'timestamp', 'raw_state', 'processed_state', 'strength',
            'ghosted', 'parent_node_id', 'ghost_nodes', 'ghost_states',
            'connections', 'last_connection_update'
        ]
        nodes = [dict(zip(columns, row)) for row in results]
        self.logger.debug(f"Loaded {len(nodes)} body_memory_nodes from DB.")
        return nodes

    # -------------------------------------------------------------------------
    # 5) COGNITIVE MEMORY CRUD
    # -------------------------------------------------------------------------
    async def create_cognitive_node(self, node_data: Dict[str, Any]) -> int:
        """
        Inserts a new record into 'cognitive_memory_nodes'.
        Returns the newly created node's ID.
        """
        query = """
            INSERT INTO cognitive_memory_nodes (
                timestamp, text_content, embedding, raw_state, processed_state,
                strength, ghosted, parent_node_id, ghost_nodes, ghost_states,
                connections, semantic_context, last_accessed, formation_source,
                last_echo_time, echo_dampening
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            node_data['timestamp'],
            node_data['text_content'],
            node_data['embedding'],
            node_data['raw_state'],
            node_data['processed_state'],
            node_data['strength'],
            node_data.get('ghosted', False),
            node_data.get('parent_node_id'),
            node_data.get('ghost_nodes', '[]'),
            node_data.get('ghost_states', '[]'),
            node_data.get('connections', '{}'),
            node_data.get('semantic_context', ''),
            node_data.get('last_accessed'),
            node_data.get('formation_source', ''),
            node_data.get('last_echo_time'),
            node_data.get('echo_dampening', 1.0)
        )
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT last_insert_rowid()")
                row = await cursor.fetchone()
                new_id = row[0]
        self.logger.debug(f"Created new cognitive_memory_node with ID {new_id}")
        return new_id

    async def get_cognitive_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Fetch a single cognitive_memory_node record by ID."""
        query = "SELECT * FROM cognitive_memory_nodes WHERE id = ?"
        results = await self._execute_query(query, (node_id,))
        if not results:
            self.logger.warning(f"No cognitive_memory_node found with ID {node_id}")
            return None

        columns = [
            'id', 'timestamp', 'text_content', 'embedding', 'raw_state', 'processed_state',
            'strength', 'ghosted', 'parent_node_id', 'ghost_nodes', 'ghost_states',
            'connections', 'semantic_context', 'last_accessed', 'formation_source',
            'last_echo_time', 'echo_dampening'
        ]
        row = results[0]
        node_dict = dict(zip(columns, row))
        self.logger.debug(f"Retrieved cognitive_memory_node: {node_dict}")
        return node_dict

    async def update_cognitive_node(self, node_id: int, node_data: Dict[str, Any]) -> None:
        """Updates an existing cognitive_memory_node by ID."""
        query = """
            UPDATE cognitive_memory_nodes
            SET timestamp = ?, text_content = ?, embedding = ?, raw_state = ?,
                processed_state = ?, strength = ?, ghosted = ?, parent_node_id = ?,
                ghost_nodes = ?, ghost_states = ?, connections = ?, semantic_context = ?,
                last_accessed = ?, formation_source = ?, last_echo_time = ?,
                echo_dampening = ?
            WHERE id = ?
        """
        params = (
            node_data['timestamp'],
            node_data['text_content'],
            node_data['embedding'],
            node_data['raw_state'],
            node_data['processed_state'],
            node_data['strength'],
            node_data.get('ghosted', False),
            node_data.get('parent_node_id'),
            node_data.get('ghost_nodes', '[]'),
            node_data.get('ghost_states', '[]'),
            node_data.get('connections', '{}'),
            node_data.get('semantic_context', ''),
            node_data.get('last_accessed'),
            node_data.get('formation_source', ''),
            node_data.get('last_echo_time'),
            node_data.get('echo_dampening', 1.0),
            node_id
        )
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
            if conn.total_changes == 0:
                raise DatabaseError(f"Cognitive node {node_id} not found for update.")
        self.logger.debug(f"Updated cognitive_memory_node {node_id}")

    async def delete_cognitive_node(self, node_id: int) -> None:
        """Removes a cognitive_memory_node by ID (plus any referencing data if needed)."""
        query = "DELETE FROM cognitive_memory_nodes WHERE id = ?"
        async with self.transaction() as conn:
            await self._execute_query(query, (node_id,), conn=conn)
            if conn.total_changes == 0:
                raise DatabaseError(f"CognitiveMemoryNode {node_id} not found for deletion.")
            # Also remove references from memory_links and synthesis_relations
            await self._execute_query(
                "DELETE FROM memory_links WHERE cognitive_node_id = ?",
                (node_id,),
                conn=conn
            )
            await self._execute_query(
                "DELETE FROM synthesis_relations WHERE synthesis_node_id = ? OR constituent_node_id = ?",
                (node_id, node_id),
                conn=conn
            )
        self.logger.debug(f"Deleted cognitive_memory_node {node_id} and related references.")

    async def load_all_cognitive_nodes(self) -> List[Dict[str, Any]]:
        """Fetch all cognitive_memory_nodes from the DB."""
        query = "SELECT * FROM cognitive_memory_nodes"
        results = await self._execute_query(query)
        columns = [
            'id', 'timestamp', 'text_content', 'embedding', 'raw_state', 'processed_state',
            'strength', 'ghosted', 'parent_node_id', 'ghost_nodes', 'ghost_states',
            'connections', 'semantic_context', 'last_accessed', 'formation_source',
            'last_echo_time', 'echo_dampening'
        ]
        nodes = [dict(zip(columns, row)) for row in results]
        self.logger.debug(f"Loaded {len(nodes)} cognitive_memory_nodes from DB.")
        return nodes

    # -------------------------------------------------------------------------
    # 6) MEMORY LINKS & SIGNATURES
    # -------------------------------------------------------------------------
    async def add_or_update_memory_link(
        self,
        cognitive_node_id: int,
        body_node_id: int,
        link_strength: float,
        link_type: str = "direct",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Insert or update a body-cognitive link in the memory_links table.
        If the link already exists, update it; otherwise, insert new.
        """
        if metadata is None:
            metadata = {}
        meta_json = json.dumps(metadata)

        query = """
            INSERT INTO memory_links
                (cognitive_node_id, body_node_id, link_strength, link_type, created_at, metadata)
            VALUES (?, ?, ?, ?, strftime('%s','now'), ?)
            ON CONFLICT(cognitive_node_id, body_node_id) DO UPDATE
            SET link_strength = excluded.link_strength,
                link_type = excluded.link_type,
                metadata = excluded.metadata
        """
        params = (cognitive_node_id, body_node_id, link_strength, link_type, meta_json)
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
        self.logger.debug(
            f"Memory link added/updated between cognitive {cognitive_node_id} "
            f"and body {body_node_id}, strength={link_strength}, type={link_type}."
        )

    async def delete_memory_link(self, cognitive_node_id: int, body_node_id: int) -> None:
        """Remove a single memory link between cognitive and body nodes."""
        query = """
            DELETE FROM memory_links
            WHERE cognitive_node_id = ? AND body_node_id = ?
        """
        async with self.transaction() as conn:
            await self._execute_query(query, (cognitive_node_id, body_node_id), conn=conn)
            if conn.total_changes == 0:
                self.logger.warning(
                    f"No link found to delete for cognitive_node={cognitive_node_id}, body_node={body_node_id}"
                )
            else:
                self.logger.debug(
                    f"Memory link deleted for cognitive_node={cognitive_node_id}, body_node={body_node_id}."
                )

    async def get_memory_links_for_body(self, body_node_id: int) -> List[Dict[str, Any]]:
        """Retrieve all memory links referencing a given body node."""
        query = """
            SELECT cognitive_node_id, body_node_id, link_strength, link_type, created_at, metadata
            FROM memory_links
            WHERE body_node_id = ?
        """
        rows = await self._execute_query(query, (body_node_id,))
        links = []
        for r in rows:
            links.append({
                'cognitive_node_id': r[0],
                'body_node_id': r[1],
                'link_strength': r[2],
                'link_type': r[3],
                'created_at': r[4],
                'metadata': json.loads(r[5]) if r[5] else {}
            })
        return links

    async def get_memory_links_for_cognitive(self, cognitive_node_id: int) -> List[Dict[str, Any]]:
        """Retrieve all memory links referencing a given cognitive node."""
        query = """
            SELECT cognitive_node_id, body_node_id, link_strength, link_type, created_at, metadata
            FROM memory_links
            WHERE cognitive_node_id = ?
        """
        rows = await self._execute_query(query, (cognitive_node_id,))
        links = []
        for r in rows:
            links.append({
                'cognitive_node_id': r[0],
                'body_node_id': r[1],
                'link_strength': r[2],
                'link_type': r[3],
                'created_at': r[4],
                'metadata': json.loads(r[5]) if r[5] else {}
            })
        return links

    async def preserve_body_signature(
        self,
        cognitive_node_id: int,
        body_node_id: int,
        signature: BodyStateSignature
    ) -> None:
        """
        Store a body state signature in the memory_links metadata.
        Merges with existing metadata if present.
        """
        if not is_dataclass(signature):
            raise ValueError("Signature must be a dataclass instance.")
        try:
            result = await self._execute_query(
                "SELECT metadata FROM memory_links "
                "WHERE cognitive_node_id = ? AND body_node_id = ?",
                (cognitive_node_id, body_node_id)
            )
            if not result:
                raise DatabaseError(
                    f"No memory link found between cognitive {cognitive_node_id} and body {body_node_id}"
                )
            existing_meta = json.loads(result[0][0]) if result[0][0] else {}
            signatures = existing_meta.get('body_signatures', [])
            signatures.append(asdict(signature))
            existing_meta['body_signatures'] = signatures

            meta_json = json.dumps(existing_meta)
            update_query = """
                UPDATE memory_links SET metadata = ?
                WHERE cognitive_node_id = ? AND body_node_id = ?
            """
            async with self.transaction() as conn:
                await self._execute_query(
                    update_query,
                    (meta_json, cognitive_node_id, body_node_id),
                    conn=conn
                )
            self.logger.debug(
                f"Preserved body signature for link (cognitive={cognitive_node_id}, body={body_node_id})."
            )
        except (json.JSONDecodeError, IndexError) as e:
            self.logger.error(f"Failed to preserve signature: {str(e)}")
            raise DatabaseError("Signature preservation failed")

    async def get_preserved_signatures(
        self,
        cognitive_node_id: int,
        body_node_id: int
    ) -> List[Dict[str, Any]]:
        """Retrieve all body state signatures from the memory_links metadata."""
        query = """
            SELECT metadata FROM memory_links
            WHERE cognitive_node_id = ? AND body_node_id = ?
        """
        result = await self._execute_query(query, (cognitive_node_id, body_node_id))
        if not result or not result[0][0]:
            return []

        try:
            meta = json.loads(result[0][0])
            return meta.get('body_signatures', [])
        except json.JSONDecodeError:
            self.logger.warning(
                f"Invalid metadata format in memory_links for cognitive={cognitive_node_id}, body={body_node_id}"
            )
            return []

    # -------------------------------------------------------------------------
    # 7) SYNTHESIS RELATIONS
    # -------------------------------------------------------------------------
    async def add_synthesis_relation(
        self,
        synthesis_node_id: int,
        constituent_node_id: int,
        relationship_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a record in 'synthesis_relations' describing how
        one cognitive node synthesizes or merges another.
        """
        if metadata is None:
            metadata = {}
        query = """
            INSERT INTO synthesis_relations
            (synthesis_node_id, constituent_node_id, relationship_type, metadata)
            VALUES (?, ?, ?, ?)
        """
        params = (
            synthesis_node_id,
            constituent_node_id,
            relationship_type,
            json.dumps(metadata)
        )
        async with self.transaction() as conn:
            await self._execute_query(query, params, conn=conn)
        self.logger.debug(
            f"Synthesis relation added. Synthesis={synthesis_node_id}, "
            f"Constituent={constituent_node_id}, Type={relationship_type}."
        )

    async def get_synthesis_relations(self, node_id: int, as_synthesis: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves all synthesis relations for a node, either as the 'synthesis_node_id'
        or as the 'constituent_node_id'.
        """
        if as_synthesis:
            query = """
                SELECT constituent_node_id, relationship_type, metadata
                FROM synthesis_relations
                WHERE synthesis_node_id = ?
            """
        else:
            query = """
                SELECT synthesis_node_id, relationship_type, metadata
                FROM synthesis_relations
                WHERE constituent_node_id = ?
            """
        rows = await self._execute_query(query, (node_id,))
        relations = []
        for r in rows:
            try:
                meta = json.loads(r[2]) if r[2] else {}
            except json.JSONDecodeError:
                meta = {}
            relations.append({
                'node_id': r[0],
                'relationship_type': r[1],
                'metadata': meta
            })
        return relations

    async def delete_synthesis_relations(self, node_id: int) -> None:
        """
        Removes all synthesis relations referencing the given node
        (whether as synthesis_node_id or constituent_node_id).
        """
        query = """
            DELETE FROM synthesis_relations
            WHERE synthesis_node_id = ? OR constituent_node_id = ?
        """
        async with self.transaction() as conn:
            await self._execute_query(query, (node_id, node_id), conn=conn)
        self.logger.debug(f"Deleted all synthesis_relations involving node {node_id}.")

    # -------------------------------------------------------------------------
    # 8) CROSS-MEMORY UTILITIES
    # -------------------------------------------------------------------------
    async def get_cognitive_references(self, body_node_id: int) -> List[int]:
        """
        Get IDs of cognitive nodes referencing a given body node
        (via memory_links).
        """
        query = """
            SELECT cognitive_node_id
            FROM memory_links
            WHERE body_node_id = ?
        """
        rows = await self._execute_query(query, (body_node_id,))
        refs = [r[0] for r in rows]
        self.logger.debug(f"Cognitive references for body_node {body_node_id}: {refs}")
        return refs

    async def get_body_references(self, cognitive_node_id: int) -> List[int]:
        """
        Get IDs of body nodes referenced by a given cognitive node
        (via memory_links).
        """
        query = """
            SELECT body_node_id
            FROM memory_links
            WHERE cognitive_node_id = ?
        """
        rows = await self._execute_query(query, (cognitive_node_id,))
        refs = [r[0] for r in rows]
        self.logger.debug(f"Body references for cognitive_node {cognitive_node_id}: {refs}")
        return refs

    async def transfer_body_links(self, from_cog_id: int, to_cog_id: int, strength_modifier: float = 0.9) -> None:
        """
        Transfer all body links from one cognitive node to another,
        optionally modifying link strength in the process.
        """
        links = await self.get_body_references(from_cog_id)
        async with self.transaction() as conn:
            for body_id in links:
                link_query = """
                    SELECT link_strength, link_type, metadata
                    FROM memory_links
                    WHERE cognitive_node_id = ? AND body_node_id = ?
                """
                row = await self._execute_query(link_query, (from_cog_id, body_id), conn=conn)
                if not row:
                    continue
                link_strength, link_type, meta_json = row[0]
                new_strength = link_strength * strength_modifier
                meta = json.loads(meta_json) if meta_json else {}
                await self.add_or_update_memory_link(
                    cognitive_node_id=to_cog_id,
                    body_node_id=body_id,
                    link_strength=new_strength,
                    link_type=link_type,
                    metadata=meta
                )
                await self._execute_query(
                    "DELETE FROM memory_links WHERE cognitive_node_id = ? AND body_node_id = ?",
                    (from_cog_id, body_id),
                    conn=conn
                )
        self.logger.debug(f"Transferred {len(links)} body links from cog {from_cog_id} to cog {to_cog_id}.")

    # -------------------------------------------------------------------------
    # 9) NODE LIFECYCLE UTILITIES
    # -------------------------------------------------------------------------
    async def get_node_ancestry(self, node_id: int, is_body_node: bool) -> List[int]:
        """
        Get the full ancestry chain for a node.
        E.g., a node's parent, parent's parent, etc.
        """
        table = 'body_memory_nodes' if is_body_node else 'cognitive_memory_nodes'
        ancestry = []
        current_id = node_id

        while current_id:
            query = f"SELECT parent_node_id FROM {table} WHERE id = ?"
            result = await self._execute_query(query, (current_id,))
            if not result:
                break

            parent_id = result[0][0]
            if parent_id and parent_id not in ancestry:
                ancestry.append(parent_id)
                current_id = parent_id
            else:
                break

        self.logger.debug(
            f"Ancestry for {'body' if is_body_node else 'cognitive'} node {node_id}: {ancestry}"
        )
        return ancestry

    async def get_ghost_cluster(self, node_id: int, is_body_node: bool) -> List[int]:
        """
        Get the list of 'ghost' nodes embedded within the specified node's ghost_nodes field.
        """
        table = 'body_memory_nodes' if is_body_node else 'cognitive_memory_nodes'
        query = f"SELECT ghost_nodes FROM {table} WHERE id = ?"
        result = await self._execute_query(query, (node_id,))
        if not result or not result[0][0]:
            self.logger.debug(f"No ghost nodes found for {table} node {node_id}")
            return []

        try:
            ghost_array = json.loads(result[0][0])
            cluster = [n['node_id'] for n in ghost_array if 'node_id' in n]
            self.logger.debug(f"Ghost cluster for node {node_id}: {cluster}")
            return cluster
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.warning(
                f"Invalid ghost_nodes format for node {node_id}: {str(e)}"
            )
            return []
