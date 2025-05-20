"""
State management and synchronization for Hephia.
Manages both persistent session state and API context distribution.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import sqlite3
import zlib

from event_dispatcher import global_event_dispatcher, Event
from internal.internal import Internal

@dataclass
class PersistentState:
    """
    Complete system state for persistence between sessions.
    Contains detailed state data needed for full system reconstruction.
    """
    session_state: Dict[str, Any]  # Full internal module states
    brain_state: Dict[str, Any]    # Brain/cognitive state
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to database-friendly format."""
        return {
            'session_state': self.session_state,
            'brain_state': self.brain_state,
            'timestamp': self.timestamp.isoformat()
        }

class StateBridge:
    """
    Manages both session persistence and real-time state distribution.
    
    Handles two distinct types of state:
    1. Session State: Detailed state for system reconstruction between sessions
    2. API Context: Processed state for real-time system communication
    """
    
    def __init__(self, internal: Optional[Internal] = None):
        self.internal = internal
        self.internal_context = internal.context if internal else None
        self.persistent_state: Optional[PersistentState] = None
        self.state_lock = asyncio.Lock()
        self.db_path = 'data/server_state.db'
        self.last_vacuum_time = datetime.min
        self.last_cleanup_time = datetime.min
        self.vacuum_interval = timedelta(minutes=10) 
        self.cleanup_interval = timedelta(minutes=5) 
        self.max_states = 5
        self.last_state_hash = None
        self.last_cognitive_summary: Optional[str] = ""

    async def initialize(self):
        """Initialize state management and restore previous session if available."""
        await self._init_database()
        await self._cleanup_old_states(force=True)  # force initial cleanup
        
        if self.internal:
            try:
                # Restore previous session if available
                loaded_state = await self._load_last_session()
                if loaded_state:
                    # Extract module states for restoration
                    session_state = {
                        'needs': loaded_state.session_state.get('needs', {}),
                        'behavior': loaded_state.session_state.get('behavior', {}),
                        'emotions': loaded_state.session_state.get('emotions', {}),
                        'mood': loaded_state.session_state.get('mood', {})
                    }
                    
                    # Restore and stabilize internal state
                    await self.internal.restore_state(session_state)
                    await self.internal.shake()
                
                # Start internal systems
                await self.internal.start()
                
                # Initialize current persistent state
                self.persistent_state = PersistentState(
                    session_state=self._collect_session_state(),
                    brain_state=loaded_state.brain_state if loaded_state else {},
                    timestamp=datetime.now()
                )

                # Set initial state hash
                self.last_state_hash = self._compute_state_hash(self.persistent_state)
                
                # Save initial state
                await self._save_session()

                self.setup_event_listeners()

                context = await self.internal_context.get_api_context(use_memory_emotions=False)
                # Notify system with API context
                global_event_dispatcher.dispatch_event(
                    Event("state:initialized", {
                        "context": context
                    })
                )
            except Exception as e:
                print(f"Error initializing state management: {e}")
                raise

    def setup_event_listeners(self):
        """Set up event listeners for cognitive state update"""
        global_event_dispatcher.add_listener(
            "cognitive:context_update",
            lambda event: asyncio.create_task(self.update_cognitive_state(event))
        )

    async def update_cognitive_state(self, event: Event):
        """Update cognitive state and broadcast API context."""
        async with self.state_lock:
            if self.persistent_state and event.data.get('source') == 'exo_processor':
                self.persistent_state.brain_state = event.data.get('raw_state', {})
                self.last_cognitive_summary = event.data.get('processed_state', "")
        try:
            await self.update_state() 
        except Exception as e:
            print(f"Error in StateBridge.update_cognitive_state after processing event: {e}")
            raise

    def _validate_brain_state(self, state: Any) -> Optional[List[Dict[str, str]]]:
        """Validate and normalize brain state structure."""
        if not isinstance(state, list):
            return None
            
        # Check each message has required structure
        for msg in state:
            if not isinstance(msg, dict):
                return None
            if 'role' not in msg or 'content' not in msg:
                return None
            if msg['role'] not in ['system', 'user', 'assistant']:
                return None
                
        return state

    def _collect_session_state(self) -> Dict[str, Any]:
        """Collect complete session state from all internal modules."""
        raw_state = {
            'needs': self.internal.needs_manager.get_needs_state(),
            'behavior': self.internal.behavior_manager.get_behavior_state(),
            'emotions': self.internal.emotional_processor.get_emotional_state(),
            'mood': self.internal.mood_synthesizer.get_mood_state()
        }
        
        # Optimize emotional vectors storage
        if 'emotions' in raw_state and 'active_vectors' in raw_state['emotions']:
            # Filter vectors by significance
            significant_vectors = [
                vector for vector in raw_state['emotions']['active_vectors']
                if vector.get('intensity', 0) > 0.1  # Only keep vectors with meaningful intensity
            ]
            # Sort by intensity and keep top N most significant
            significant_vectors.sort(key=lambda x: x.get('intensity', 0), reverse=True)
            raw_state['emotions']['active_vectors'] = significant_vectors[:100]  # Limit to 100 most significant vectors
            
        return raw_state
    
    def _compute_state_hash(self, state: PersistentState) -> str:
        """Compute a hash of the significant parts of state to detect meaningful changes."""
        # Focus on key state elements that would constitute a meaningful change
        key_elements = {
            'emotions': state.session_state.get('emotions', {}),
            'needs': state.session_state.get('needs', {}),
            'mood': state.session_state.get('mood', {}),
            'behavior': state.session_state.get('behavior', {})
        }
        return str(hash(json.dumps(key_elements, sort_keys=True)))

    async def get_api_context(self, use_memory_emotions: bool = True) -> Dict[str, Any]:
        """Get current processed state for API consumption."""
        if self.internal:
            return await self.internal_context.get_api_context(use_memory_emotions=use_memory_emotions)
        return {}
    
    def get_latest_cognitive_summary(self) -> str: # Return str, as default is ""
        """Returns the most recently cached cognitive summary (processed_state)."""
        return self.last_cognitive_summary if self.last_cognitive_summary is not None else ""
    
    def get_latest_raw_conversation_state(self) -> List[Dict[str, str]]:
        """Returns the most recently cached raw conversation state (brain_state)."""
        if self.persistent_state and self.persistent_state.brain_state is not None:
            return self.persistent_state.brain_state
        return []

    async def update_state(self):
        """Update both persistent state and broadcast API context."""
        async with self.state_lock:
            try:
                # Update persistent state
                new_state = PersistentState(
                    session_state=self._collect_session_state(),
                    brain_state=self.persistent_state.brain_state if self.persistent_state else {},
                    timestamp=datetime.now()
                )

                # Compute new state hash
                new_hash = self._compute_state_hash(new_state)
                
                # Only announce & save if state has changed significantly
                if new_hash != self.last_state_hash:
                    self.persistent_state = new_state
                    self.last_state_hash = new_hash
                    await self._save_session()

                    # Cleanup old states periodically
                    if datetime.now() - self.last_cleanup_time >= self.cleanup_interval:
                        await self._cleanup_old_states()
                        self.last_cleanup_time = datetime.now()

                    # Broadcast API context
                    context = await self.internal_context.get_api_context(use_memory_emotions=False)
                    global_event_dispatcher.dispatch_event(
                        Event("state:changed", {
                            "context": context
                        })
                    )
            except Exception as e:
                print(f"Error updating state: {e}")
                raise
    
    # Database operations
    async def _init_database(self):
        """Initialize the state database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable auto_vacuum mode (if not already set).
        cursor.execute("PRAGMA auto_vacuum")
        mode = cursor.fetchone()[0]
        if mode == 0:
            cursor.execute("PRAGMA auto_vacuum = FULL")
            print("Auto-vacuum set to FULL.")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_state BLOB,
                brain_state TEXT,
                compressed BOOLEAN DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    async def _save_session(self):
        if not self.persistent_state:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            session_json = json.dumps(self.persistent_state.session_state)
            compressed_session = zlib.compress(session_json.encode())
            cursor.execute("""
                INSERT INTO system_state (timestamp, session_state, brain_state, compressed)
                VALUES (?, ?, ?, ?)
            """, (
                self.persistent_state.timestamp.isoformat(),
                compressed_session,
                json.dumps(self.persistent_state.brain_state),
                True
            ))
            conn.commit()
        finally:
            conn.close()

    async def _load_last_session(self) -> Optional[PersistentState]:
        """Load most recent session state."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT timestamp, session_state, brain_state, compressed
                FROM system_state
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                timestamp, session_state, brain_state, compressed = row
                
                # Decompress session state if needed
                if compressed:
                    session_state = zlib.decompress(session_state).decode()
                
                brain_state = json.loads(brain_state)
                session_state = json.loads(session_state)
                
                # Validate brain state structure
                validated_brain_state = self._validate_brain_state(brain_state)
                
                return PersistentState(
                    session_state=session_state,
                    brain_state=validated_brain_state or [], 
                    timestamp=datetime.fromisoformat(timestamp)
                )
        finally:
            conn.close()
        
        return None

    async def _cleanup_old_states(self, force: bool = False):
        """
        Delete old state entries, keeping only the most recent self.max_states entries.
        Optionally, force cleanup regardless of the cleanup interval.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            # Delete states older than the most recent self.max_states
            cursor.execute(f"""
                DELETE FROM system_state 
                WHERE id NOT IN (
                    SELECT id FROM system_state 
                    ORDER BY timestamp DESC 
                    LIMIT {self.max_states}
                )
            """)
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old state entries")
            conn.commit()

            current_time = datetime.now()
            if force or current_time - self.last_vacuum_time >= self.vacuum_interval:
                cursor.execute("VACUUM")
                self.last_vacuum_time = current_time
                print("Database VACUUM completed")
        finally:
            conn.close()