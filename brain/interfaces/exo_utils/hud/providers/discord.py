# brain/interfaces/exo_utils/hud/providers/discord.py

from typing import Dict, Any, List, Optional

from .base import BaseHudProvider
from brain.prompting.loader import get_prompt
from core.discord_service import DiscordService
from config import Config
from loggers import BrainLogger

class DiscordHudProvider(BaseHudProvider):
    """
    HUD Provider for Discord-related information.
    Displays active channel, recent messages, and user summary.
    """
    
    # extra timeout given that it has to actually fetch data from Discord
    PROVIDER_TIMEOUT_SECONDS: float = 2.0

    def __init__(self, discord_service: DiscordService, prompt_key: str = 'interface.exo.hud.discord', section_name: str = "Discord"):
        super().__init__(prompt_key=prompt_key, section_name=section_name)
        if not discord_service:
            raise ValueError("DiscordService instance is required for DiscordHudProvider.")
        self.discord_service = discord_service

    async def _prepare_prompt_vars(self, hud_metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Fetches and prepares pre-formatted strings for the Discord HUD section,
        suitable for direct substitution via string.Template.
        Keys in the returned dictionary will correspond to placeholders in the '*.combined' prompt.
        """
        discord_vars = {
            "hud_header_str": f"[{self.section_name}]",
            "discord_channel_path": "",
            "discord_messages_block_str": "",
            "discord_users_summary_str": "",
            "discord_error_str": "",
            "discord_is_active_for_hud_str": "false"
        }

        if not Config.get_discord_enabled():
            discord_vars["discord_error_str"] = "Discord is currently disabled by System Config"
            return discord_vars
        
        discord_vars["discord_is_active_for_hud_str"] = "true"

        last_channel_path = hud_metadata.get("last_discord_channel_path")
        if not last_channel_path:
            discord_vars["discord_error_message"] = "No active channel."
            return discord_vars

        discord_vars["discord_channel_path"] = last_channel_path
        
        error_messages_for_hud_list = []

        try:
            hud_message_limit = 10
            max_snippet_len = 500

            history_data, hist_status = await self.discord_service.get_enhanced_history(
                last_channel_path, limit=hud_message_limit
            )

            if hist_status == 200 and history_data and history_data.get("messages"):
                message_lines = ["  Recent Messages:"]
                for msg in history_data.get("messages"):
                    try:
                        ref = msg.get('reference', '')
                        timestamp = msg.get('timestamp', '')[:16]  # YYYY-MM-DD HH:MM
                        author = msg.get('author', 'Unknown')
                        content = msg.get('content', '').replace('\n', ' ')
                        if len(content) > max_snippet_len:  # Truncate very long messages
                            content = content[:max_snippet_len] + "..."
                        
                        # Include reference ID for easy referencing in commands
                        message_lines.append(f"[{timestamp}] {ref} {author}: {content}")
                    except Exception as e:
                        BrainLogger.error(f"HUD: Error processing message: {e}", exc_info=True)

                discord_vars["discord_messages_block_str"] = "\n".join(message_lines)
            elif hist_status is not None and hist_status >= 400:
                err_msg_part = f"History (Status {hist_status})"
                if isinstance(history_data, dict) and history_data.get("error"): err_msg_part += f": {history_data['error']}"
                error_messages_for_hud_list.append(err_msg_part)
                discord_vars["discord_messages_block_str"] = f"  Recent Messages: Unavailable ({err_msg_part.split(':')[0].strip()})"
            else: # Success but empty message list or other non-error case
                 discord_vars["discord_messages_block_str"] = "  Recent Messages: None found."
        except Exception as e:
            BrainLogger.error(f"HUD ({self.section_name}): Exc. in history for {last_channel_path}: {e}", exc_info=True)
            error_messages_for_hud_list.append("History (Processing Error)")
            discord_vars["discord_messages_block_str"] = "  Recent Messages: Error retrieving."

        # --- Fetch and Format Active Users ---
        try:
            users_data, users_status = await self.discord_service.get_user_list(last_channel_path)
            if users_status == 200 and users_data:
                recent_users = [f"{user['display_name']}" for user in users_data.get('recently_active', [])]
                other_users = [f"{user['display_name']}" for user in users_data.get('other_members', [])]

                num_recent = len(recent_users)
                num_other = len(other_users)
                recent_list_str = ", ".join(recent_users) if recent_users else "None"
                other_list_str = ", ".join(other_users) if other_users else "None"
                
                # reuse for now; make specific to HUD later
                discord_vars["discord_users_summary_str"] = get_prompt("interfaces.exo.hud.discord.users",
                    model=Config.get_cognitive_model(),
                    vars={
                        "num_recent": num_recent,
                        "num_other": num_other,
                        "recent_list": recent_list_str,
                        "other_list": other_list_str
                    }
                )
            elif users_status is not None and users_status >= 400:
                err_msg_part = f"Users (Status {users_status})"
                if isinstance(users_data, dict) and users_data.get("error"): err_msg_part += f": {users_data['error']}"
                error_messages_for_hud_list.append(err_msg_part)
                discord_vars["discord_users_summary_str"] = f"  Active Users: Unavailable ({err_msg_part.split(':')[0].strip()})"
            else: # Success but empty
                discord_vars["discord_users_summary_str"] = "  Active Users: None present"

        except Exception as e:
            BrainLogger.error(f"HUD ({self.section_name}): Exc. in users for {last_channel_path}: {e}", exc_info=True)
            error_messages_for_hud_list.append("Users (Processing Error)")
            discord_vars["discord_users_summary_str"] = "  Active Users: Error retrieving."

        if error_messages_for_hud_list:
            discord_vars["discord_error_str"] = f"  Info: {'; '.join(error_messages_for_hud_list)}"
        
        return discord_vars