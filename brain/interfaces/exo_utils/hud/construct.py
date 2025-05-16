# brain/interfaces/exo_utils/hud/construct.py

import asyncio
from typing import Dict, Any, List, Optional

from core.discord_service import DiscordService

from loggers import BrainLogger
from config import Config

from .sections.base import BaseHudSection
from .sections.discord import DiscordHudSection
from .sections.system import SystemHudSection
from .sections.internals import InternalStateHudSection
# from .sections.goals_section import GoalsHudSection # Future
# from .sections.pins_section import PinsHudSection # Future
# from .sections.notes_section import NotesHudSection # Future

class HudConstructor:
    """
    Constructs the complete HUD overlay string by orchestrating various HUD data sections.
    """
    def __init__(
        self,
        # Services needed by various sections will be passed here
        discord_service: DiscordService,
        # notes_manager: Optional[NotesManager] = None, # Future
        # Add other global services sections might need
    ):
        self.sections: List[BaseHudSection] = []
        self._initialize_sections(
            discord_service=discord_service,
            # notes_manager=notes_manager
        )

    def _initialize_sections(
        self,
        discord_service: DiscordService,
        # notes_manager: Optional[NotesManager]
    ):
        """
        Initializes and registers all active HUD sections based on configuration and available services.
        """
        # System Section
        self.sections.append(SystemHudSection(prompt_key='interfaces.exo.hud.system', section_name="System"))

        # Internal State Section
        self.sections.append(InternalStateHudSection(prompt_key='interfaces.exo.hud.internals', section_name="Internal State"))

        # Discord Section (conditional on config)
        if Config.get_discord_enabled():
            self.sections.append(DiscordHudSection(discord_service, prompt_key='interfaces.exo.hud.discord', section_name="Discord"))
        else:
            BrainLogger.warning("HUD: Discord not enabled in config, Discord HUD section disabled.")

        BrainLogger.info(f"HUD: Initialized {len(self.sections)} sections.")


    async def build_hud_string(self, hud_metadata: Dict[str, Any], current_model_name: str) -> str:
        """
        Builds the complete HUD string by concurrently fetching and rendering data
        from all registered and active sections.

        Args:
            hud_metadata: Shared dynamic metadata (e.g., last_channel_path, last_interaction_timestamp).
            current_model_name: The name of the current LLM for prompt selection.

        Returns:
            A single string representing the complete HUD overlay.
        """
        if not self.sections:
            return "[HUD: No sections configured]"

        tasks = [
            section.get_rendered_section_string(hud_metadata, current_model_name)
            for section in self.sections
        ]
        
        # Execute all section tasks concurrently
        # return_exceptions=True allows us to handle individual section failures
        rendered_sections_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_hud_parts = []
        for result in rendered_sections_results:
            if isinstance(result, Exception):
                BrainLogger.error(f"HUD: Uncaught exception from a section during gather: {result}", exc_info=result)
                final_hud_parts.append("[HUD: Section Processing Error]") 
            elif result and result.strip(): # Add if the string is not None and not empty/whitespace
                final_hud_parts.append(result)
        
        final_hud_string = "\n".join(final_hud_parts)
        
        return final_hud_string if final_hud_string.strip() else "[HUD: Context unavailable or initializing...]"