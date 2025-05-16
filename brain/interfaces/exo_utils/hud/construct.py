# brain/interfaces/exo_utils/hud/construct.py

import asyncio
from typing import Dict, Any, List, Optional

from core.discord_service import DiscordService

from loggers import BrainLogger
from config import Config

from .providers.base import BaseHudProvider
from .providers.discord import DiscordHudProvider
from .providers.system import SystemHudProvider
from .providers.internals import InternalStateHudProvider
# from .providers.goals_provider import GoalsHudProvider # Future
# from .providers.pins_provider import PinsHudProvider # Future
# from .providers.notes_provider import NotesHudProvider # Future

class HudConstructor:
    """
    Constructs the complete HUD overlay string by orchestrating various HUD data providers.
    """
    def __init__(
        self,
        # Services needed by various providers will be passed here
        discord_service: DiscordService,
        # notes_manager: Optional[NotesManager] = None, # Future
        # Add other global services providers might need
    ):
        self.providers: List[BaseHudProvider] = []
        self._initialize_providers(
            discord_service=discord_service,
            # notes_manager=notes_manager
        )

    def _initialize_providers(
        self,
        discord_service: DiscordService,
        # notes_manager: Optional[NotesManager]
    ):
        """
        Initializes and registers all active HUD providers based on configuration and available services.
        """
        # System Provider
        self.providers.append(SystemHudProvider(prompt_key='interfaces.exo.hud.system', section_name="System"))

        # Internal State Provider
        self.providers.append(InternalStateHudProvider(prompt_key='interfaces.exo.hud.internals', section_name="Internal State"))

        # Discord Provider (conditional on config)
        if Config.get_discord_enabled():
            self.providers.append(DiscordHudProvider(discord_service, prompt_key='interfaces.exo.hud.discord', section_name="Discord"))
        else:
            BrainLogger.warning("HUD: Discord not enabled in config, Discord HUD section disabled.")

        BrainLogger.info(f"HUD: Initialized {len(self.providers)} providers.")


    async def build_hud_string(self, hud_metadata: Dict[str, Any], current_model_name: str) -> str:
        """
        Builds the complete HUD string by concurrently fetching and rendering data
        from all registered and active providers.

        Args:
            hud_metadata: Shared dynamic metadata (e.g., last_channel_path, last_interaction_timestamp).
            current_model_name: The name of the current LLM for prompt selection.

        Returns:
            A single string representing the complete HUD overlay.
        """
        if not self.providers:
            return "[HUD: No providers configured]"

        tasks = [
            provider.get_rendered_section_string(hud_metadata, current_model_name)
            for provider in self.providers
        ]
        
        # Execute all provider tasks concurrently
        # return_exceptions=True allows us to handle individual provider failures
        rendered_sections_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_hud_parts = []
        for result in rendered_sections_results:
            if isinstance(result, Exception):
                BrainLogger.error(f"HUD: Uncaught exception from a provider during gather: {result}", exc_info=result)
                final_hud_parts.append("[HUD: Section Processing Error]") 
            elif result and result.strip(): # Add if the string is not None and not empty/whitespace
                final_hud_parts.append(result)
        
        final_hud_string = "\n".join(final_hud_parts)
        
        return final_hud_string if final_hud_string.strip() else "[HUD: Context unavailable or initializing...]"