"""
web.py - Web page access and interaction environment.

Focused on accessing and viewing web content, distinct from
search queries or note-taking functionality.
"""

import aiohttp
from typing import Dict, List, Any
from .base_environment import BaseEnvironment
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult
)

class WebEnvironment(BaseEnvironment):
    """
    Web environment for accessing and viewing web pages.
    Provides direct URL access and content viewing capabilities.
    """
    
    def __init__(self):
        super().__init__()
        
        self.help_text = """
        The web environment allows you to access and view web pages.
        Use 'open' to view page content directly.
        Content can be saved to notes or explored further with search.
        
        Examples:
        - web open https://example.com --preview=full
        - web open https://news.site/article --extract=text
        """
    
    def _register_commands(self) -> None:
        """Register web environment commands."""
        self.register_command(
            CommandDefinition(
                name="open",
                description="Access and view a web page",
                parameters=[
                    Parameter(
                        name="url",
                        description="The URL to access (including https://)",
                        required=True,
                        examples=[
                            "https://example.com",
                            "https://news.website/article"
                        ]
                    )
                ],
                flags=[
                    Flag(
                        name="preview",
                        description="Content preview length",
                        type=ParameterType.STRING,
                        default="medium",
                        examples=["--preview=short", "--preview=full"]
                    ),
                    Flag(
                        name="extract",
                        description="Content extraction mode",
                        type=ParameterType.STRING,
                        default="default",
                        examples=["--extract=text", "--extract=links"]
                    )
                ],
                examples=[
                    "web open https://example.com",
                    "web open https://example.com --preview=full",
                    "web open https://news.site --extract=text"
                ],
                related_commands=[
                    'notes create "Save web content"',
                    'search query "Related topic"'
                ],
                failure_hints={
                    "404": "Page not found. Check the URL is correct.",
                    "ConnectionError": "Could not connect to website. Check the URL or try again.",
                    "SSLError": "Secure connection failed. Make sure the URL starts with https://"
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
        """Execute web commands."""
        if action == "open":
            url = params[0]
            preview_length = flags.get("preview", "medium")
            extract_mode = flags.get("extract", "default")
            
            # Validate URL format
            if not url.startswith(("http://", "https://")):
                return CommandResult(
                    success=False,
                    message="URL must start with http:// or https://",
                    suggested_commands=[f"web open https://{url}"],
                    error="Invalid URL format"
                )
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            text = await response.text()
                            
                            # Handle preview length
                            preview_limits = {
                                "short": 500,
                                "medium": 1000,
                                "full": None
                            }
                            limit = preview_limits.get(preview_length, 1000)
                            
                            # Basic content processing based on extract mode
                            if extract_mode == "text":
                                # Simple text extraction - could be enhanced
                                content = " ".join(text.split())
                            elif extract_mode == "links":
                                # Simple link extraction - could be enhanced
                                import re
                                links = re.findall(r'href="(https?://[^"]+)"', text)
                                content = "\n".join(links[:10])  # First 10 links
                            else:
                                content = text
                            
                            if limit:
                                content = content[:limit] + "..." if len(content) > limit else content
                            
                            return CommandResult(
                                success=True,
                                message=f"Content from {url}:\n\n{content}",
                                suggested_commands=[
                                    f'notes create "Content from {url}"',
                                    f'search query "About {url.split("/")[-1]}"',
                                    f'web open {url} --preview=full'
                                ],
                                data={
                                    "url": url,
                                    "preview_length": preview_length,
                                    "extract_mode": extract_mode
                                }
                            )
                        else:
                            return CommandResult(
                                success=False,
                                message=f"Failed to fetch page: Status {response.status}",
                                suggested_commands=["web open <different_url>"],
                                error=f"HTTP {response.status}"
                            )
                            
            except aiohttp.ClientError as e:
                return CommandResult(
                    success=False,
                    message=f"Failed to access URL: {str(e)}",
                    suggested_commands=["web open <different_url>"],
                    error=str(e)
                )