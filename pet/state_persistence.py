"""
Pet state persistence system.
Handles saving and loading of core module states while maintaining extensibility.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

@dataclass
class NeedState:
    """Core need state."""
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)  # For future extensions

@dataclass
class EmotionalVectorState:
    """Emotional vector state with extensible properties."""
    valence: float
    arousal: float
    intensity: float
    timestamp: float
    source_type: Optional[str] = None
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BehaviorState:
    """Behavior system state."""
    current_behavior: str
    locked_until: Optional[float] = None
    locked_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryState:
    """Memory system state with emotional history."""
    emotional_log: List[EmotionalVectorState]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorePetState:
    """
    Complete pet internal state.
    Each component has its own section and a metadata dict for future extensions.
    """
    needs: Dict[str, NeedState]
    emotional_vectors: List[EmotionalVectorState]
    behavior: BehaviorState
    memory: MemoryState
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"  

class PetStateManager:
    """Manages saving and loading of pet internal state."""

    def __init__(self, state_dir: str = 'data/internal_state'):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / 'core_state.json'
        self.backup_dir = self.state_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)

    def save_state(self, pet) -> None:
        """
        Save current pet state.
        Creates both current and backup states.
        """
        try:
            state = self._extract_state(pet)
            
            # Save current state
            self._write_state(self.state_file, state)
            
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f'state_backup_{timestamp}.json'
            self._write_state(backup_file, state)
            
            # Cleanup old backups (keep last 5)
            self._cleanup_old_backups()
            
        except Exception as e:
            print(f"Error saving pet state: {e}")
            raise

    def load_state(self) -> Optional[CorePetState]:
        """
        Load most recent valid state.
        Falls back to backups if main state is corrupted.
        """
        try:
            # Try main state file
            if self.state_file.exists():
                try:
                    return self._read_state(self.state_file)
                except Exception as e:
                    print(f"Error reading main state, trying backup: {e}")

            # Try latest backup
            backups = sorted(self.backup_dir.glob('state_backup_*.json'))
            if backups:
                return self._read_state(backups[-1])
                
            return None
            
        except Exception as e:
            print(f"Error loading pet state: {e}")
            return None

    def _extract_state(self, pet) -> CorePetState:
        """Extract core state from pet instance."""
        return CorePetState(
            needs={
                name: NeedState(
                    value=need.value,
                    metadata=getattr(need, 'metadata', {})
                )
                for name, need in pet.needs_manager.needs.items()
            },
            emotional_vectors=[
                EmotionalVectorState(
                    valence=v.valence,
                    arousal=v.arousal,
                    intensity=v.intensity,
                    timestamp=v.timestamp,
                    source_type=v.source_type,
                    name=v.name,
                    metadata=getattr(v, 'metadata', {})
                )
                for v in pet.emotional_processor.current_stimulus.active_vectors
            ],
            behavior=BehaviorState(
                current_behavior=pet.behavior_manager.current_behavior.name,
                locked_until=pet.behavior_manager.locked_until,
                locked_by=pet.behavior_manager.locked_by,
                metadata={}
            ),
            memory=MemoryState(
                emotional_log=[
                    EmotionalVectorState(
                        valence=e.valence,
                        arousal=e.arousal,
                        intensity=e.intensity,
                        timestamp=e.timestamp,
                        source_type=e.source_type,
                        name=e.name,
                        metadata=getattr(e, 'metadata', {})
                    )
                    for e in pet.memory_system.body_memory.emotional_log
                ],
                metadata={}
            ),
            timestamp=datetime.now()
        )

    def _write_state(self, path: Path, state: CorePetState) -> None:
        """Write state to file with proper formatting."""
        with open(path, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)

    def _read_state(self, path: Path) -> CorePetState:
        """Read and validate state from file."""
        with open(path, 'r') as f:
            data = json.load(f)
            
        # Convert timestamp string back to datetime
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        return CorePetState(**data)

    def _cleanup_old_backups(self, keep: int = 5) -> None:
        """Maintain only recent backups."""
        backups = sorted(self.backup_dir.glob('state_backup_*.json'))
        for backup in backups[:-keep]:
            backup.unlink()