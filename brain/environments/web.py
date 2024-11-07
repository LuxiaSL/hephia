"""
Simple web environment for basic page access.
"""

from typing import Dict, Any, Optional
import aiohttp
from brain.environments.base_environment import BaseEnvironment

class WebEnvironment(BaseEnvironment):
    """Basic web access environment."""
    
    def get_commands(self) -> list:
        """Get available web commands."""
        return [
            {
                "name": "open",
                "description": "Open a URL and view page contents"
            },
            {
                "name": "help",
                "description": "Show Web environment help"
            }
        ]
    
    async def handle_command(self, command: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Handle web commands."""
        parts = command.split(maxsplit=1)
        action = parts[0].lower()
        
        if action == "open" and len(parts) > 1:
            return await self._open_url(parts[1].strip())
        elif action == "help":
            return self.format_help()
        else:
            return {
                "title": "Error",
                "content": f"Unknown web command: {action}"
            }
    
    async def _open_url(self, url: str) -> Dict[str, str]:
        """
        Fetch and return page content.
        Basic implementation - can be enhanced with proper scraping later.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Basic content truncation for now
                        content = text[:1000] + "..." if len(text) > 1000 else text
                        
                        return {
                            "title": f"Web Page: {url}",
                            "content": f"Content preview:\n\n{content}\n\n---\nUse 'web open <url>' to visit another page."
                        }
                    else:
                        return {
                            "title": "Error",
                            "content": f"Failed to fetch page: Status {response.status}"
                        }
        except Exception as e:
            return {
                "title": "Error",
                "content": f"Failed to access URL: {str(e)}"
            }

    def format_help(self) -> Dict[str, str]:
        """Format help text."""
        return {
            "title": "Web Environment Help",
            "content": """Available commands:
- web open <url> : Open a URL and view its contents
- web help : Show this help message

Example:
web open https://example.com"""
        }