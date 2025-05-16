# brain/interfaces/exo_utils/hud/sections/base.py

from abc import ABC, abstractmethod
import asyncio
from typing import Dict, Any

from brain.prompting.loader import get_prompt # Assuming direct import is fine
from loggers import BrainLogger # For consistent logging

class BaseHudSection(ABC):
    """
    Abstract base class for all HUD data sections.
    Each section is responsible for fetching data for its section of the HUD,
    preparing variables for its prompt template, and rendering the template.
    """

    # Cooldown or update interval for this section (in seconds)
    # 0 means update every time. Can be overridden by subclasses.
    # This is a placeholder for a more sophisticated update mechanism if needed.
    # For now, we'll assume most update every cycle unless they implement their own check.
    DEFAULT_UPDATE_INTERVAL: float = 0.0 
    # Timeout for this specific section's data fetching/rendering.
    SECTION_TIMEOUT_SECONDS: float = 1.0 

    def __init__(self, prompt_key: str, section_name: str = "DefaultSection"):
        """
        Args:
            prompt_key: The key used to retrieve the prompt template (e.g., 'hud.datetime').
            section_name: A human-readable name for this section's section (for logging/errors).
        """
        if not prompt_key:
            raise ValueError("Prompt key cannot be empty for a HUD section.")
        self.prompt_key = prompt_key
        self.section_name = section_name
        self._last_update_time: float = 0.0 # For future use with update intervals

    @abstractmethod
    async def _prepare_prompt_vars(self, hud_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Fetches and prepares the necessary data for this section's HUD section.
        This method must be implemented by subclasses.

        Args:
            hud_metadata: Shared metadata from ExoProcessorInterface (e.g., last_channel_path).

        Returns:
            A dictionary of string variables to be substituted into the prompt template.
            If the section should not be rendered (e.g., feature disabled, no relevant data),
            it can return an empty dict or a dict that leads to an empty render.
        """
        pass

    async def get_rendered_section_string(self, hud_metadata: Dict[str, Any], current_model_name: str) -> str:
        """
        Gets the data, prepares variables, and renders the prompt for this HUD section.
        Handles timeouts and errors gracefully.

        Args:
            hud_metadata: Shared metadata from ExoProcessorInterface.
            current_model_name: The name of the current LLM model for prompt selection.

        Returns:
            The rendered string for this HUD section, or a fallback error/status string.
            Returns an empty string if the section should not be rendered.
        """
        # Placeholder for future update interval logic:
        # current_time = time.time()
        # if self.DEFAULT_UPDATE_INTERVAL > 0 and \
        #    (current_time - self._last_update_time) < self.DEFAULT_UPDATE_INTERVAL and \
        #    self._cached_render:
        #     return self._cached_render

        fallback_error_string = f"[HUD: {self.section_name} - Unavailable]"
        rendered_string = ""

        try:
            # Applying timeout to the critical data preparation step
            async with asyncio.timeout(self.SECTION_TIMEOUT_SECONDS):
                prompt_vars = await self._prepare_prompt_vars(hud_metadata)

            # Only proceed to render if prompt_vars suggest rendering is needed.
            # sections can return a specific key like 'is_active_for_hud': False
            # or ensure prompt_vars is empty if nothing should be rendered.
            # For now, we assume if prompt_vars is not None/empty, we try to render.
            if prompt_vars is None: # section explicitly decided not to render
                return ""

            # Render the prompt using the prepared variables
            # The get_prompt function is stateless and handles its own caching.
            rendered_string = get_prompt(key=f"{self.prompt_key}.combined", model=current_model_name, vars=prompt_vars)
            
            # self._last_update_time = current_time # For future caching
            # self._cached_render = rendered_string # For future caching

            # Ensure we return empty string if rendered_string is None or only whitespace
            return rendered_string if rendered_string and rendered_string.strip() else ""

        except asyncio.TimeoutError:
            BrainLogger.warning(f"HUD: Timeout in {self.section_name} section after {self.SECTION_TIMEOUT_SECONDS}s.")
            return f"[HUD: {self.section_name} - Timeout]"
        except FileNotFoundError as e: # Specifically for prompt file issues
            BrainLogger.error(f"HUD: Prompt file not found for {self.section_name} ({self.prompt_key}): {e}", exc_info=True)
            return fallback_error_string
        except KeyError as e: # Specifically for missing keys in prompt YAML or vars
            BrainLogger.error(f"HUD: Key error rendering prompt for {self.section_name} ({self.prompt_key}): {e}", exc_info=True)
            return fallback_error_string
        except Exception as e:
            BrainLogger.error(f"HUD: Error in {self.section_name} section: {e}", exc_info=True)
            return fallback_error_string # Generic fallback for other errors