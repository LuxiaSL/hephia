"""
Simple web environment for basic page access.
"""

from typing import Dict, Any, Optional
import aiohttp
from brain.environments.base_environment import BaseEnvironment
from brain.environments.terminal_formatter import TerminalFormatter, CommandResponse, EnvironmentHelp

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
    
    async def handle_command(self, command: str) -> CommandResponse:
        """Handle web commands."""
        parts = command.split(maxsplit=1)
        action = parts[0].lower()
        
        if action == "open" and len(parts) > 1:
            return await self._open_url(parts[1].strip())
        elif action == "help":
            return self.format_help()
        else:
            return CommandResponse(
                title="Error",
                content=f"Unknown web command: {action}",
                suggested_commands=["web help", "web open <url>"]
            )
    
    async def _open_url(self, url: str) -> CommandResponse:
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
                        
                        return CommandResponse(
                            title=f"Web Page: {url}",
                            content=f"Content preview:\n\n{content}",
                            suggested_commands=[
                                "web open <another_url>",
                                f'notes create "Web content from {url}"',
                                "search query <related_topic>"
                            ]
                        )
                    else:
                        return CommandResponse(
                            title="Error",
                            content=f"Failed to fetch page: Status {response.status}",
                            suggested_commands=["web help"]
                        )
        except Exception as e:
            return CommandResponse(
                title="Error",
                content=f"Failed to access URL: {str(e)}",
                suggested_commands=["web help"]
            )

    def format_help(self) -> CommandResponse:
        return TerminalFormatter.format_environment_help(
            EnvironmentHelp(
                name="Web",
                commands=self.get_commands(),
                examples=[
                    "web open https://example.com",
                    "web open https://news.website/article"
                ],
                tips=[
                    "Use full URLs including https://",
                    "Save interesting content using notes",
                    "Follow up on web content with search queries"
                ]
            )
        )