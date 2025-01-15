"""
State management and synchronization for Hephia.
Manages both persistent session state and API context distribution.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import os
import sqlite3

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
    
    async def initialize(self):
        """Initialize state management and restore previous session if available."""
        await self._init_database()
        await self._cleanup_old_states()
        
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
                
                # Save initial state
                await self._save_session()

                self.setup_event_listeners()
                
                # Notify system with API context
                global_event_dispatcher.dispatch_event(
                    Event("state:initialized", {
                        "context": self.internal_context.get_api_context()
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
            try:
                # Update cognitive state
                self.persistent_state.brain_state = event.data.get('raw_state', {})
                await self.update_state()
            except Exception as e:
                print(f"Error updating cognitive state: {e}")
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
        return {
            'needs': self.internal.needs_manager.get_needs_state(),
            'behavior': self.internal.behavior_manager.get_behavior_state(),
            'emotions': self.internal.emotional_processor.get_emotional_state(),
            'mood': self.internal.mood_synthesizer.get_mood_state()
        }

    def get_api_context(self) -> Dict[str, Any]:
        """Get current processed state for API consumption."""
        if self.internal:
            return self.internal_context.get_api_context()
        return {}

    async def update_state(self):
        """Update both persistent state and broadcast API context."""
        async with self.state_lock:
            try:
                # Update persistent state
                self.persistent_state = PersistentState(
                    session_state=self._collect_session_state(),
                    brain_state=self.persistent_state.brain_state if self.persistent_state else {},
                    timestamp=datetime.now()
                )
                
                # Save to database
                await self._save_session()
                
                # Broadcast API context
                global_event_dispatcher.dispatch_event(
                    Event("state:changed", {
                        "context": self.internal_context.get_api_context()
                    })
                )
            except Exception as e:
                print(f"Error updating state: {e}")
                raise
    
    async def process_timer_event(self, event: Event):
        """Handle timer-triggered state updates."""
        if self.internal:
            try:
                await self.update_state()
            except Exception as e:
                print(f"Error processing timer event: {e}")
    
    # Database operations
    async def _init_database(self):
        """Initialize the state database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                session_state TEXT,
                brain_state TEXT
            )
        """)
        conn.commit()
        conn.close()

    async def _save_session(self):
        """Save current session state to database."""
        if not self.persistent_state:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO system_state (timestamp, session_state, brain_state)
                VALUES (?, ?, ?)
            """, (
                self.persistent_state.timestamp.isoformat(),
                json.dumps(self.persistent_state.session_state),
                json.dumps(self.persistent_state.brain_state)
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
                SELECT timestamp, session_state, brain_state
                FROM system_state
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                timestamp, session_state, brain_state = row
                brain_state = json.loads(brain_state)
                
                # Validate brain state structure
                validated_brain_state = self._validate_brain_state(brain_state)
                
                return PersistentState(
                    session_state=json.loads(session_state),
                    brain_state=validated_brain_state or [], 
                    timestamp=datetime.fromisoformat(timestamp)
                )
        finally:
            conn.close()
        
        return None

    async def _cleanup_old_states(self):
        """Clean up old state entries from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                DELETE FROM system_state 
                WHERE id NOT IN (
                    SELECT id 
                    FROM system_state 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                )
            """)
            
            if cursor.rowcount > 0:
                print(f"Cleaned up {cursor.rowcount} old state entries")
            conn.commit()
        finally:
            conn.close()