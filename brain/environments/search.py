"""
Simple search environment using Perplexity API.
"""

from typing import Dict, Any
from api_clients import APIManager
from brain.environments.base_environment import BaseEnvironment

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
    
    async def handle_command(self, command: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Handle search commands."""
        parts = command.split(maxsplit=1)
        action = parts[0].lower()
        
        if action == "query" and len(parts) > 1:
            return await self._perform_search(parts[1])
        elif action == "help":
            return self.format_help()
        else:
            return {
                "title": "Error",
                "content": f"Unknown search command: {action}"
            }
    
    async def _perform_search(self, query: str) -> Dict[str, str]:
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
            
            return {
                "title": "Search Results",
                "content": response["choices"][0]["message"]["content"] + "\n\n" +
                          "Available actions:\n" +
                          "• 'notes create' to save insights\n" +
                          "• 'search query' to explore further"
            }
            
        except Exception as e:
            return {
                "title": "Search Error",
                "content": f"Failed to perform search: {str(e)}"
            }
    
    def format_help(self) -> Dict[str, str]:
        """Format help text."""
        return {
            "title": "Search Help",
            "content": """╔══════════════════════════╗
║     SEARCH COMMANDS     ║
╚══════════════════════════╝

• search query <terms> - Search and summarize information

Example:
search query "current weather in London"

Tips:
- Be specific in your queries
- Use notes to save interesting findings"""
        }