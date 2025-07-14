# brain/interfaces/exo_utils/hud/sections/stickys.py

import sqlite3
import os
from typing import Dict, Any, List, Optional
from config import Config
from loggers import BrainLogger
from .base import BaseHudSection

class StickyNotesHudSection(BaseHudSection):
    """
    HUD section for displaying sticky notes in the system prompt.
    Queries the notes database directly for maximum simplicity and defensive programming.
    """

    def __init__(self, prompt_key: str = 'interfaces.exo.hud.stickys', section_name: str = "Sticky Notes"):
        """
        Args:
            prompt_key: Key for the YAML template
            section_name: Display name for logging
        """
        super().__init__(prompt_key, section_name)
        self.db_path = 'data/notes.db'

    async def _get_sticky_notes(self) -> Optional[List[Dict[str, Any]]]:
        """
        Directly query the database for sticky notes.
        
        Returns:
            List of sticky note dictionaries or None if database unavailable
        """
        try:
            # Check if database exists
            if not os.path.exists(self.db_path):
                BrainLogger.debug("HUD: Notes database doesn't exist yet")
                return None

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if sticky column exists (defensive against old database schema)
            cursor.execute("PRAGMA table_info(notes)")
            columns = {col[1] for col in cursor.fetchall()}
            
            if 'sticky' not in columns:
                BrainLogger.debug("HUD: Notes database doesn't have sticky column yet")
                conn.close()
                return None

            # Query for sticky notes
            cursor.execute("""
                SELECT llm_id, content, tags, created_at
                FROM notes 
                WHERE sticky = 1 
                ORDER BY created_at DESC
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "id": row[0] or "unknown",
                    "content": row[1] or "",
                    "tags": row[2].split(',') if row[2] else [],
                    "created": row[3] or ""
                }
                for row in results
            ]
            
        except sqlite3.Error as e:
            BrainLogger.warning(f"HUD: Database error fetching sticky notes: {e}")
            return None
        except Exception as e:
            BrainLogger.error(f"HUD: Unexpected error fetching sticky notes: {e}", exc_info=True)
            return None

    async def _prepare_prompt_vars(self, hud_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Fetch sticky notes and prepare formatted strings for the HUD template.
        Following the Discord pattern: do all logic in Python, return formatted strings.
        
        Returns:
            Dictionary of pre-formatted strings for template substitution
        """
        sticky_vars = {
            "hud_header_str": f"[{self.section_name}]",
            "sticky_notes_block_str": "",
            "sticky_error_str": "",
            "sticky_is_active_for_hud_str": "false"
        }

        try:
            sticky_notes = await self._get_sticky_notes()
            
            # Case 1: Database not available/accessible
            if sticky_notes is None:
                sticky_vars["sticky_error_str"] = "Notes system initializing..."
                return sticky_vars
            
            # Case 2: No sticky notes exist
            if not sticky_notes:
                sticky_vars["sticky_notes_block_str"] = "No sticky notes set. Use for goals, reminders, persistent context.\nTry: notes create \"Remember this important goal\" --sticky=true"
                sticky_vars["sticky_is_active_for_hud_str"] = "true"
                return sticky_vars
            
            # Case 3: Sticky notes exist - format them
            sticky_vars["sticky_is_active_for_hud_str"] = "true"
            
            formatted_notes = []
            for note in sticky_notes:
                content = note['content']

                # Format tags if they exist
                tags_str = f" #{','.join(note['tags'])}" if note['tags'] else ""
                formatted_notes.append(f"  📌 {note['id']}: {content}{tags_str}")
            
            # Add header with count
            count = len(sticky_notes)
            max_sticky = Config.MAX_STICKY_NOTES
            header = f"sticky notes ({count}/{max_sticky}):"
            
            all_lines = [header] + formatted_notes
            sticky_vars["sticky_notes_block_str"] = "\n".join(all_lines)
            
            return sticky_vars

        except Exception as e:
            BrainLogger.error(f"HUD: Error preparing sticky notes vars: {e}", exc_info=True)
            sticky_vars["sticky_error_str"] = "Error loading sticky notes"
            return sticky_vars