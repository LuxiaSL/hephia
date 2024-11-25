"""
State management and synchronization for Hephia.
Handles state persistence, updates, and synchronization across all systems.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import os
import sqlite3

from event_dispatcher import global_event_dispatcher, Event
from pet.pet import Pet

@dataclass
class SystemState:
    """Represents the complete system state."""
    pet_state: Dict[str, Any]
    brain_state: Dict[str, Any]
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary format."""
        return {
            'pet_state': self.pet_state,
            'brain_state': self.brain_state,
            'last_updated': self.last_updated.isoformat()
        }

class StateBridge:
    """Manages state synchronization between all system components."""
    
    def __init__(self, pet: Optional[Pet] = None):
        self.pet = pet
        self.pet_context = pet.context if pet else None
        self.current_state: Optional[SystemState] = None
        self.state_lock = asyncio.Lock()
        self.db_path = 'data/hephia_state.db'
    
    async def initialize(self):
        """Initialize the state bridge and set up database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_state (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                pet_state TEXT,
                brain_state TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Clean up old states
        await self._periodic_state_cleanup()
        
        # Initialize with fresh API context
        if self.pet_context:
            try:
                self.current_state = SystemState(
                    pet_state=self.pet_context.get_api_context(),
                    brain_state={},
                    last_updated=datetime.now()
                )
                
                # Validate initial state
                if not await self._validate_state_consistency():
                    await self._handle_state_inconsistency()
                else:
                    global_event_dispatcher.dispatch_event(
                        Event("state:initialized", self.current_state.to_dict())
                    )
                    
            except Exception as e:
                print(f"Error initializing state: {e}")
    
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current state in API-friendly format."""
        async with self.state_lock:
            try:
                if self.pet:
                    current_state = {
                        'pet_state': self.pet_context.get_api_context(),
                        'brain_state': {},
                        'last_updated': datetime.now().isoformat()
                    }
                    return current_state
                else:
                    return self.current_state.to_dict() if self.current_state else {
                        'pet_state': {},
                        'brain_state': {},
                        'last_updated': datetime.now().isoformat()
                    }
            except Exception as e:
                print(f"Error getting current state: {e}")
                raise
    
    async def update_state(self, pet_state: Optional[Dict] = None, brain_state: Optional[Dict] = None):
        """Update system state and notify listeners."""
        async with self.state_lock:
            try:
                if not self.current_state:
                    return
                
                # Always get fresh API context if we have a pet
                if self.pet_context:
                    self.current_state.pet_state = self.pet_context.get_api_context()
                elif pet_state:
                    self.current_state.pet_state.update(pet_state)
                
                if brain_state:
                    self.current_state.brain_state.update(brain_state)
                
                self.current_state.last_updated = datetime.now()
                
                # Notify state change
                state_dict = self.current_state.to_dict()
                global_event_dispatcher.dispatch_event(
                    Event("state:changed", state_dict)
                )
                
                await self.save_state()
            except Exception as e:
                print(f"Error updating state: {e}")
                raise
    
    async def process_timer_event(self, event: Event):
        """Handle timer-triggered state updates."""
        if self.pet:
            try:
                # Always use fresh API context for timer updates
                await self.update_state()
            except Exception as e:
                print(f"Error processing timer event: {e}")
    
    async def save_state(self):
        """Save current state to database."""
        if not self.current_state:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO system_state (timestamp, pet_state, brain_state)
                VALUES (?, ?, ?)
            """, (
                self.current_state.last_updated.isoformat(),
                json.dumps(self.current_state.pet_state),
                json.dumps(self.current_state.brain_state)
            ))
            conn.commit()
        except Exception as e:
            print(f"Error saving state: {e}")
            raise
        finally:
            conn.close()
    
    async def load_last_state(self) -> Optional[SystemState]:
        """Load most recent state from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT timestamp, pet_state, brain_state
                FROM system_state
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                timestamp, pet_state, brain_state = row
                return SystemState(
                    pet_state=json.loads(pet_state),
                    brain_state=json.loads(brain_state),
                    last_updated=datetime.fromisoformat(timestamp)
                )
        finally:
            conn.close()
        
        return None

    async def _validate_state_consistency(self) -> bool:
        """
        Validate that stored state matches current pet state.
        Used for debugging and ensuring state synchronization.
        
        Returns:
            bool: True if states match, False otherwise
        """
        if not self.pet:
            return True
            
        try:
            current_pet_state = self.pet_context.get_api_context()
            stored_pet_state = self.current_state.pet_state if self.current_state else {}
            
            # Basic structure check
            if not stored_pet_state:
                return False
                
            # Check core state components exist and match format
            required_keys = ['mood', 'needs', 'behavior', 'emotions']
            for key in required_keys:
                if key not in stored_pet_state or key not in current_pet_state:
                    print(f"Missing required key: {key}")
                    return False
                
                # Check basic structure of each component
                if type(stored_pet_state[key]) != type(current_pet_state[key]):
                    print(f"Type mismatch for {key}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error validating state consistency: {e}")
            return False
        
    async def _handle_state_inconsistency(self):
        """
        Handle cases where stored state doesn't match current pet state.
        Attempts to reconcile differences or triggers a state reset if necessary.
        """
        if not await self._validate_state_consistency():
            try:
                print("State inconsistency detected, attempting reset...")
                
                # Get fresh state from pet
                new_state = SystemState(
                    pet_state=self.pet_context.get_api_context(),
                    brain_state=self.current_state.brain_state if self.current_state else {},
                    last_updated=datetime.now()
                )
                
                self.current_state = new_state
                await self.save_state()
                
                # Notify system of state reset
                global_event_dispatcher.dispatch_event(
                    Event("state:reset", {
                        "reason": "inconsistency_detected",
                        "new_state": self.current_state.to_dict()
                    })
                )
                
                print("State reset complete")
                
            except Exception as e:
                print(f"Error handling state inconsistency: {e}")
            
    async def _periodic_state_cleanup(self):
        """
        Periodically clean up old state entries from the database.
        Keeps the most recent states and removes older ones.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Keep only the last 100 states
            cursor.execute("""
                DELETE FROM system_state 
                WHERE id NOT IN (
                    SELECT id 
                    FROM system_state 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                )
            """)
            
            rows_deleted = cursor.rowcount
            conn.commit()
            
            if rows_deleted > 0:
                print(f"Cleaned up {rows_deleted} old state entries")
                
        except Exception as e:
            print(f"Error during state cleanup: {e}")
        finally:
            conn.close()