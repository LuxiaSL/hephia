"""
services/settings_service.py

Runtime settings with JSON file persistence.
Propagates model/threshold changes to Config and PetCognitiveBridge.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, TYPE_CHECKING

from shared_models.api_models import PetSettings, SettingsResponse
from config import Config
from loggers import SystemLogger

if TYPE_CHECKING:
    from internal.modules.cognition.cognitive_bridge import PetCognitiveBridge


class SettingsService:
    """
    Runtime settings manager.
    Loads from / saves to data/settings.json.
    Propagates changes to Config env vars and bridge thresholds.
    """

    SETTINGS_PATH = "data/settings.json"

    def __init__(self, bridge: Optional[PetCognitiveBridge] = None) -> None:
        self.bridge = bridge
        self.settings = self._load()

    def get(self) -> SettingsResponse:
        """Return current settings plus read-only extras."""
        return SettingsResponse(
            settings=self.settings,
            available_models=list(Config.AVAILABLE_MODELS.keys()),
            version=Config.VERSION,
        )

    def update(self, new_settings: PetSettings) -> SettingsResponse:
        """Validate, apply, persist, and return updated settings."""
        # Validate models exist
        if new_settings.pet_model not in Config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown pet_model: {new_settings.pet_model}")
        if new_settings.worker_model not in Config.AVAILABLE_MODELS:
            raise ValueError(f"Unknown worker_model: {new_settings.worker_model}")

        # Propagate model changes via env vars so Config.get_*_model() picks them up
        if new_settings.pet_model != self.settings.pet_model:
            os.environ["PET_MODEL"] = new_settings.pet_model
            SystemLogger.info(f"Pet model changed to: {new_settings.pet_model}")

        if new_settings.worker_model != self.settings.worker_model:
            os.environ["WORKER_MODEL"] = new_settings.worker_model
            SystemLogger.info(f"Worker model changed to: {new_settings.worker_model}")

        # Propagate threshold changes
        if (
            new_settings.memory_significance_threshold
            != self.settings.memory_significance_threshold
        ):
            if self.bridge is not None:
                self.bridge.SIGNIFICANCE_THRESHOLD = (
                    new_settings.memory_significance_threshold
                )
            SystemLogger.info(
                f"Memory significance threshold changed to: "
                f"{new_settings.memory_significance_threshold}"
            )

        # Propagate max conversation turns
        if new_settings.max_conversation_turns != self.settings.max_conversation_turns:
            os.environ["EXO_MAX_TURNS"] = str(new_settings.max_conversation_turns)
            SystemLogger.info(
                f"Max conversation turns changed to: {new_settings.max_conversation_turns}"
            )

        self.settings = new_settings
        self._save()

        return self.get()

    def _load(self) -> PetSettings:
        """Load settings from JSON file, falling back to defaults."""
        try:
            if os.path.exists(self.SETTINGS_PATH):
                with open(self.SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return PetSettings(**data)
        except Exception as e:
            SystemLogger.warning(f"Failed to load settings, using defaults: {e}")
        return PetSettings()

    def _save(self) -> None:
        """Persist current settings to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_PATH) or ".", exist_ok=True)
            with open(self.SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.settings.model_dump(), f, indent=2)
        except Exception as e:
            SystemLogger.error(f"Failed to save settings: {e}")
