"""
search.py - Information retrieval environment using Perplexity API.

Focused on direct information queries and summaries to minimize
overlap with web browsing or note-taking functionality.
"""

from typing import Dict, Any, List
from api_clients import APIManager
from .base_environment import BaseEnvironment
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult,
    CommandValidationError
)
from config import Config

class SearchEnvironment(BaseEnvironment):
    """
    Search environment for retrieving and summarizing information.
    Provides direct answers and summaries via Perplexity API.
    """
    
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        super().__init__()
        
        # Extended help for environment
        self.help_text = """
        The search environment lets you find information and get summarized answers.
        Use 'query' for direct questions and information retrieval.
        Results can be saved to notes or expanded with follow-up searches.
        
        Examples:
        - search query "current weather in London"
        - search query "explain quantum computing" --detail=high
        - search query "latest news about AI" --limit=3
        """
    
    def _register_commands(self) -> None:
        """Register search environment commands."""
        self.register_command(
            CommandDefinition(
                name="query",
                description="Search for and summarize information",
                parameters=[
                    Parameter(
                        name="search_terms",
                        description="What to search for",
                        required=True,
                        examples=[
                            '"current weather in London"',
                            '"explain quantum computing"',
                            '"latest news about AI"'
                        ]
                    )
                ],
                flags=[
                    Flag(
                        name="detail",
                        description="Level of detail in the response",
                        type=ParameterType.STRING,
                        default="medium",
                        examples=["--detail=high", "--detail=low"]
                    ),
                    Flag(
                        name="limit",
                        description="Maximum number of results/points",
                        type=ParameterType.INTEGER,
                        default=5,
                        examples=["--limit=3", "--limit=10"]
                    )
                ],
                examples=[
                    'search query "current weather in London"',
                    'search query "explain quantum computing" --detail=high',
                    'search query "latest news about AI" --limit=3'
                ],
                related_commands=[
                    'query "follow up question"',
                    'notes create "Save interesting findings"'
                ],
                failure_hints={
                    "APIError": "Search service temporarily unavailable. Try again in a moment.",
                    "RateLimitError": "Too many searches. Please wait a moment."
                }
            )
        )
    
    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """Execute search commands."""
        if action == "query":
            search_terms = params[0]
            detail = flags.get("detail", "medium")
            limit = flags.get("limit", 5)
            
            try:
                # Construct system prompt based on detail level
                detail_prompts = {
                    "low": "Provide a brief, focused summary of key points.",
                    "medium": "Provide a clear, balanced summary with main details.",
                    "high": "Provide a comprehensive summary with detailed explanations."
                }
                
                if detail not in detail_prompts:
                    detail = "medium"  
                
                messages = [
                    {
                        "role": "system",
                        "content": f"{detail_prompts[detail]} Limit to {limit} key points."
                    },
                    {
                        "role": "user",
                        "content": search_terms.strip()
                    }
                ]
                
                model_config = Config.AVAILABLE_MODELS["perplexity"]

                # Call Perplexity API via APIManager
                content = await self.api.create_completion(
                    provider=model_config.provider.value,
                    model=model_config.model_id,
                    messages=messages,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens,
                    return_content_only=True 
                )
                
                # Generate suggested follow-ups
                suggested = [
                    f'notes create "Research: {search_terms[:30]}..."',
                ]
                
                # Add detail toggle if not already used
                if detail != "high":
                    suggested.append(f'search query "{search_terms}" --detail=high')
                
                return CommandResult(
                    success=True,
                    message=content,
                    suggested_commands=suggested,
                    data={
                        "query": search_terms,
                        "detail_level": detail,
                        "limit": limit
                    }
                )
                
            except Exception as e:
                error = CommandValidationError(
                    message=str(e),
                    suggested_fixes=[
                        "Try rephrasing your search terms",
                        "Check if search service is available",
                        "Try reducing detail level or result limit"
                    ],
                    related_commands=[
                        'search query "simpler terms"',
                        'search query "..." --detail=low',
                        'help'
                    ],
                    examples=[
                        'search query "basic explanation of quantum computing"',
                        'search query "weather" --detail=low --limit=3'
                    ]
                )
                
                return CommandResult(
                    success=False,
                    message=f"Search failed: {str(e)}",
                    suggested_commands=[
                        'search query "simpler terms"',
                        'notes search',
                        'help'
                    ],
                    error=error
                )
