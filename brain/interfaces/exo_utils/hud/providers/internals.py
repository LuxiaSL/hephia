# brain/interfaces/exo_utils/hud/providers/internals.py

import time
from typing import Dict, Any

from .base import BaseHudProvider
from loggers import BrainLogger

class InternalStateHudProvider(BaseHudProvider):
    """
    HUD Provider for the agent's internal cognitive and emotional state,
    and relevant memories.
    """

    def __init__(self, prompt_key: str = 'interfaces.exo.hud.internals', section_name: str = "Internal State"):
        super().__init__(prompt_key=prompt_key, section_name=section_name)

    def _format_relative_time(self, timestamp: float) -> str:
        """Helper to format memory timestamps"""
        current_time_unix = time.time()
        time_diff = current_time_unix - timestamp
        
        if time_diff < 300: return "Just now"
        if time_diff < 3600: return "Recently"
        if time_diff < 86400: return "Earlier today"
        if time_diff < 172800: return "Yesterday"
        if time_diff < 604800: return "A few days ago"
        if time_diff < 2592000: return "A while ago"
        return "A long time ago"


    async def _prepare_prompt_vars(self, hud_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepares variables for the internal state and memories HUD section.
        Data is expected to be in hud_metadata['state_block'] and hud_metadata['memories_block'].
        """
        state_vars = {
            "hud_header_str": f"[{self.section_name}]",
            # Core State
            "internal_state_mood_str": "Mood: N/A",
            "internal_state_behavior_str": "Behavior: N/A",
            "internal_state_needs_summary_str": "Needs: N/A", # Summary line for needs
            "internal_state_emotions_summary_str": "Emotions: N/A", # Summary line for emotions

            # Memories
            "internal_state_memories_block_str": "", # Multi-line block for memories, or "  Memories: None"
            "internal_state_has_memories_str": "false", # "true" or "false"
            
            # Error/Status
            "internal_state_error_str": ""
        }

        state_block = hud_metadata.get('state_block')
        memories_block = hud_metadata.get('memories_block')

        if not state_block:
            state_vars["internal_state_error_str"] = "State data not available."
            # Still return the vars dict so the prompt can handle it gracefully
            return state_vars

        # 1. Process Core State from state_block
        try:
            # Mood
            mood_data = state_block.get('mood', {})
            mood_name = mood_data.get('name', 'Neutral')
            valence = mood_data.get('valence', 0.0)
            arousal = mood_data.get('arousal', 0.0)
            state_vars["internal_state_mood_str"] = f"{mood_name} mood (valence: {valence:.2f}, arousal: {arousal:.2f})"

            # Behavior
            behavior_data = state_block.get('behavior', {})
            behavior_name = behavior_data.get('name', 'idle')
            state_vars["internal_state_behavior_str"] = f"{behavior_name} behavior"

            # Needs
            needs_data = state_block.get('needs', {})
            needs_summary_parts = []
            sorted_needs = sorted(
                [(k, v.get('satisfaction', 1.0)) for k, v in needs_data.items() if isinstance(v, dict)], 
                key=lambda item: item[1]
            ) # Sort by satisfaction, lowest first
            
            for need_name, satisfaction_val in sorted_needs:
                urgency = "high" if satisfaction_val < 0.35 else "moderate" if satisfaction_val < 0.75 else "low"
                needs_summary_parts.append(f"{need_name}: {satisfaction_val*100:.1f}% ({urgency} urgency)")
            
            if needs_summary_parts:
                state_vars["internal_state_needs_summary_str"] = f"needs: {', '.join(needs_summary_parts)}"
            else:
                state_vars["internal_state_needs_summary_str"] = "needs: stable"

            # Emotional State - create a concise summary
            emotions_data = state_block.get('emotional_state', [])
            if emotions_data:
                # Sort by intensity, highest first
                sorted_emotions = sorted(emotions_data, key=lambda e: e.get('intensity', 0.0), reverse=True)
                emotion_summary_parts = [
                    f"{e.get('name', 'Unknown')} (strength: {e.get('intensity', 0.0):.2f})"
                    for e in sorted_emotions
                ]
                state_vars["internal_state_emotions_summary_str"] = f"emotions: {', '.join(emotion_summary_parts)}"
            else:
                state_vars["internal_state_emotions_summary_str"] = "emotions: neutral"

        except Exception as e:
            BrainLogger.error(f"HUD ({self.section_name}): Error processing state_block: {e}", exc_info=True)
            state_vars["internal_state_error_str"] = "Error processing internal state."
            # Set defaults for state strings if processing failed mid-way
            state_vars["internal_state_mood_str"] = state_vars.get("internal_state_mood_str", "Mood: Error")
            state_vars["internal_state_behavior_str"] = state_vars.get("internal_state_behavior_str", "Behavior: Error")
            state_vars["internal_state_needs_summary_str"] = state_vars.get("internal_state_needs_summary_str", "Needs: Error")
            state_vars["internal_state_emotions_summary_str"] = state_vars.get("internal_state_emotions_summary_str", "Emotions: Error")


        # 2. Process Memories from memories_block
        try:
            if memories_block and isinstance(memories_block, list):
                memory_lines_for_hud = ["relevant memories:"]
                for memory_data in memories_block:
                    timestamp_unix = memory_data.get('timestamp')
                    time_str = self._format_relative_time(timestamp_unix) if timestamp_unix else "Unknown time"
                    
                    text_content = memory_data.get('content', memory_data.get('text_content', 'No content.'))
                    memory_lines_for_hud.append(f"  [{time_str}]")
                    for line in text_content.split('\n'):
                        if line.strip():
                            memory_lines_for_hud.append(f"  {line.strip()}")
                    memory_lines_for_hud.append("")
                
                if len(memory_lines_for_hud) > 1:
                    state_vars["internal_state_memories_block_str"] = "\n".join(memory_lines_for_hud)
                    state_vars["internal_state_has_memories_str"] = "true"
                else:
                    state_vars["internal_state_memories_block_str"] = "Relevant Memories: None recent."
                    state_vars["internal_state_has_memories_str"] = "false"
            else:
                state_vars["internal_state_memories_block_str"] = "Relevant Memories: None available."
                state_vars["internal_state_has_memories_str"] = "false"

        except Exception as e:
            BrainLogger.error(f"HUD ({self.section_name}): Error processing memories_block: {e}", exc_info=True)
            if state_vars["internal_state_error_str"]: # Append to existing error
                 state_vars["internal_state_error_str"] += "; Error processing memories."
            else:
                state_vars["internal_state_error_str"] = "Error processing memories."
            state_vars["internal_state_memories_block_str"] = "Relevant Memories: Error."
            state_vars["internal_state_has_memories_str"] = "false" # Treat as no memories on error

        return state_vars