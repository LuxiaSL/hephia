"""
Simple search environment using Perplexity API.
"""

from typing import Dict, Any
from api_clients import APIManager
from brain.environments.base_environment import BaseEnvironment
from brain.environments.terminal_formatter import TerminalFormatter, CommandResponse, EnvironmentHelp

class SearchEnvironment(BaseEnvironment):
    """Basic search functionality using Perplexity."""
    
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
    
    def get_commands(self) -> list:
        """Get available search commands."""
        return [
            {
                "name": "query",
                "description": "Search and summarize information"
            },
            {
                "name": "help",
                "description": "Show Search help"
            }
        ]
    
    async def handle_command(self, command: str) -> CommandResponse:
        """Handle search commands."""
        parts = command.split(maxsplit=1)
        action = parts[0].lower()
        
        if action == "query" and len(parts) > 1:
            return await self._perform_search(parts[1])
        elif action == "help":
            return self.format_help()
        else:
            return CommandResponse(
                title="Error",
                content=f"Unknown search command: {action}",
                suggested_commands=["search help", "search query <terms>"]
            )
    
    async def _perform_search(self, query: str) -> CommandResponse:
        """Perform search using Perplexity API."""
        try:
            response = await self.api.perplexity.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "Provide a clear, concise summary of search results. Focus on factual information."
                    },
                    {
                        "role": "user",
                        "content": query.strip()
                    }
                ]
            )
            
            return CommandResponse(
                title="Search Results",
                content=response["choices"][0]["message"]["content"],
                suggested_commands=[
                    f'notes create "Research: {query[:30]}..."',
                    f'search query "{query} more details"'
                ]
            )
            
        except Exception as e:
            return CommandResponse(
                title="Search Error",
                content=f"Failed to perform search: {str(e)}",
                suggested_commands=["search help"]
            )
    
    def format_help(self) -> CommandResponse:
        """Format help text."""
        return TerminalFormatter.format_environment_help(
            EnvironmentHelp(
                name="Search",
                commands=self.get_commands(),
                examples=[
                    'search query "current weather in London"',
                    'search query "latest news about AI"'
                ],
                tips=[
                    "Be specific in your queries",
                    "Use notes to save interesting findings",
                    "Try following up on search results with more specific queries"
                ]
            )
        )