"""
web.py - Web page access and interaction environment.

Focused on accessing and viewing web content, distinct from
search queries or note-taking functionality.
"""

import aiohttp
from bs4 import BeautifulSoup
import re

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from .base_environment import BaseEnvironment
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult,
    CommandValidationError
)

@dataclass
class ProcessedContent:
    """Structured representation of processed web content."""
    title: str
    description: Optional[str]
    main_content: str
    metadata: Dict[str, Any]
    links: List[Dict[str, str]]
    content_type: str

class WebEnvironment(BaseEnvironment):
    """
    Web environment for accessing and viewing web pages.
    Provides direct URL access and content viewing capabilities.
    """
    
    def __init__(self):
        super().__init__()
        
        self.preview_limits = {
            "short": 500,
            "medium": 1000,
            "full": 50000
        }

        self.help_text = """
        The web environment provides smart web page access and viewing.

        Features:
        - Intelligent content extraction
        - Multiple viewing modes
        - Link discovery and analysis
        - Metadata extraction

        Examples:
        - web open https://example.com --preview=full
        - web open https://news.site/article --extract=text
        - web open https://docs.python.org --extract=links
        """
    
    def _register_commands(self) -> None:
        """Register web environment commands."""
        self.register_command(
            CommandDefinition(
                name="open",
                description="Access and intelligently process a web page",
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
                        description="Content preview length (short/medium/full)",
                        type=ParameterType.STRING,
                        default="medium",
                        examples=["--preview=short", "--preview=full"]
                    ),
                    Flag(
                        name="extract",
                        description="Content extraction mode (text/links/structured)",
                        type=ParameterType.STRING,
                        default="structured",
                        examples=["--extract=text", "--extract=links"]
                    ),
                    Flag(
                        name="format",
                        description="Output format (plain/markdown)",
                        type=ParameterType.STRING,
                        default="markdown",
                        examples=["--format=plain", "--format=markdown"]
                    )
                ],
                examples=[
                    "web open https://sitetoaccess.com",
                    "web open https://sitetoaccess.com --preview=full --extract=structured",
                    "web open https://sitetoaccess.com --extract=text --format=markdown"
                ],
                related_commands=[
                    'notes create "<Summary of page>"',
                    'search query "<Related topic>"',
                ],
                failure_hints={
                    "404": "Page not found. Check the URL is correct.",
                    "ConnectionError": "Could not connect to website. Check the URL or try again.",
                    "SSLError": "Secure connection failed. Make sure the URL starts with https://",
                    "ContentTypeError": "Unsupported content type. This environment handles text/html content.",
                    "ExtractionError": "Could not extract content. The page might be dynamic or protected."
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
            extract_mode = flags.get("extract", "structured")
            output_format = flags.get("format", "markdown")
            
            # Validate URL format and structure
            if not self._validate_url(url):
                return CommandResult(
                    success=False,
                    message="Invalid URL format",
                    suggested_commands=[
                        f"web open https://{url}" if not url.startswith('http') else None,
                        "web open https://example.com"
                    ],
                    error=CommandValidationError(
                        message="URL must be properly formatted",
                        suggested_fixes=[
                            "Add https:// prefix",
                            "Check for valid domain structure"
                        ],
                        related_commands=["web open https://example.com"],
                        examples=[
                            "web open https://example.com",
                            "web open https://news.site/article"
                        ]
                    )
                )
            
            try:
                # Fetch and process content
                processed = await self._fetch_and_process_content(
                    url, preview_length, extract_mode
                )
                
                if not processed:
                    error = CommandValidationError(
                        message="Could not process page content",
                        suggested_fixes=[
                            "Check if the URL is accessible",
                            "Verify the page contains readable content",
                            "Try a different URL"
                        ],
                        related_commands=[
                            "web open https://example.com",
                            "search query <topic>"
                        ],
                        examples=[
                            "web open https://docs.python.org",
                            "web open https://news.site/article --preview=short"
                        ]
                    )
                    return CommandResult(
                        success=False,
                        message="Failed to process page content",
                        suggested_commands=[
                            "web open https://example.com",
                            "search query 'related topic'",
                            "help"
                        ],
                        error=error
                    )
                
                # Format the response based on mode and format
                formatted_content = self._format_content(
                    processed, extract_mode, output_format
                )
                
                # Generate contextual suggested commands
                suggested = self._generate_suggestions(url, processed)
                
                return CommandResult(
                    success=True,
                    message=formatted_content,
                    suggested_commands=suggested,
                    data={
                        "url": url,
                        "title": processed.title,
                        "content_type": processed.content_type,
                        "metadata": processed.metadata
                    }
                )
                
            except aiohttp.ClientError as e:
                error = CommandValidationError(
                    message=f"Failed to access URL: {str(e)}",
                    suggested_fixes=[
                        "Check if the URL is correct",
                        "Verify the site is accessible",
                        "Try using https:// instead of http://"
                    ],
                    related_commands=[
                        "web open https://example.com",
                        "search query <topic>"
                    ],
                    examples=[
                        "web open https://docs.python.org",
                        "web open https://news.site/article"
                    ]
                )
                
                suggestion = "web open <different_url>"
                if "ssl" in str(e).lower():
                    suggestion = f"web open https://{url.replace('http://', '')}"
                elif "dns" in str(e).lower():
                    suggestion = "Check domain spelling"
                
                return CommandResult(
                    success=False,
                    message=f"Failed to access URL: {str(e)}",
                    suggested_commands=[suggestion, "help"],
                    error=error
                )

            except Exception as e:
                error = CommandValidationError(
                    message=f"An unexpected error occurred: {str(e)}",
                    suggested_fixes=[
                        "Try the operation again",
                        "Verify the URL format and accessibility"
                    ],
                    related_commands=[
                        "web open https://example.com",
                        "help"
                    ],
                    examples=[
                        "web open https://docs.python.org"
                    ]
                )
                
                return CommandResult(
                    success=False,
                    message=f"Error processing request: {str(e)}",
                    suggested_commands=["help"],
                    error=error
                )
            
    def _validate_url(self, url: str) -> bool:
        """Validate URL format and structure."""
        url = url.strip('"\'')
    
        if not url.startswith(("http://", "https://")):
            return False
            
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    async def _fetch_and_process_content(
        self,
        url: str,
        preview_length: str,
        extract_mode: str
    ) -> Optional[ProcessedContent]:
        """Fetch and process web content with smart extraction."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return None
                
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' not in content_type.lower():
                    return None
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                # Extract key content
                title = self._extract_title(soup)
                description = self._extract_description(soup)
                main_content = self._extract_main_content(soup)
                metadata = self._extract_metadata(soup, response.headers)
                links = self._extract_links(soup, url)
                
                # Apply length limit
                limit = self.preview_limits.get(preview_length, 1000)
                if limit:
                    main_content = main_content[:limit] + "..." if len(main_content) > limit else main_content
                
                return ProcessedContent(
                    title=title,
                    description=description,
                    main_content=main_content,
                    metadata=metadata,
                    links=links,
                    content_type=content_type
                )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title with fallbacks."""
        # Try OpenGraph title first
        og_title = soup.find("meta", property="og:title")
        if og_title:
            return og_title["content"]
        
        # Try standard title tag
        title_tag = soup.title
        if title_tag:
            return title_tag.string.strip()
        
        # Try first h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return "Untitled Page"
    
    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page description with fallbacks."""
        # Try meta description
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            return meta_desc["content"]
        
        # Try OpenGraph description
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            return og_desc["content"]
        
        # Try first paragraph
        first_p = soup.find("p")
        if first_p:
            return first_p.get_text(strip=True)
        
        return None
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content with noise removal."""
        # Remove noisy elements
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Try to find main content area
        main = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article'))
        
        if main:
            content = main
        else:
            content = soup
        
        # Get text with some structure preservation
        lines = []
        for element in content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = element.get_text(strip=True)
            if text:  # Skip empty lines
                if element.name.startswith('h'):
                    lines.append(f"\n## {text}\n")
                else:
                    lines.append(text)
        
        return "\n".join(lines)
    
    def _extract_metadata(self, soup: BeautifulSoup, headers: Dict) -> Dict[str, Any]:
        """Extract useful metadata from page."""
        metadata = {
            "last_modified": headers.get("Last-Modified"),
            "language": soup.html.get("lang"),
            "author": None,
            "published_date": None,
            "tags": [],
            "word_count": 0
        }
        
        # Try to find author
        author = (
            soup.find("meta", {"name": "author"}) or
            soup.find("meta", property="article:author")
        )
        if author:
            metadata["author"] = author.get("content")
        
        # Try to find publication date
        pub_date = (
            soup.find("meta", property="article:published_time") or
            soup.find("meta", {"name": "date"})
        )
        if pub_date:
            metadata["published_date"] = pub_date.get("content")
        
        # Try to find keywords/tags
        keywords = (
            soup.find("meta", {"name": "keywords"}) or
            soup.find("meta", property="article:tag")
        )
        if keywords:
            metadata["tags"] = [
                tag.strip() 
                for tag in keywords.get("content", "").split(",")
                if tag.strip()
            ]
        
        # Calculate approximate word count
        text_content = soup.get_text()
        metadata["word_count"] = len(text_content.split())
        
        return metadata    
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract and process relevant links."""
        links = []
        seen_urls = set()
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            
            # Skip empty or javascript links
            if not text or href.startswith(('javascript:', '#')):
                continue
            
            # Resolve relative URLs
            full_url = urljoin(base_url, href)
            
            # Skip duplicates
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            
            links.append({
                "url": full_url,
                "text": text,
                "is_external": not full_url.startswith(base_url)
            })
        
        return links[:10]  # Limit to 10 most relevant links
    
    def _format_content(
        self,
        content: ProcessedContent,
        extract_mode: str,
        output_format: str
    ) -> str:
        """Format processed content based on mode and format."""
        if extract_mode == "text":
            if output_format == "markdown":
                return (
                    f"# {content.title}\n\n"
                    f"{content.description}\n\n"
                    f"{content.main_content}"
                )
            else:
                return (
                    f"Title: {content.title}\n\n"
                    f"Description: {content.description}\n\n"
                    f"Content:\n{content.main_content}"
                )
                
        elif extract_mode == "links":
            if output_format == "markdown":
                links_text = [
                    f"- [{link['text']}]({link['url']})"
                    f"{' (External)' if link['is_external'] else ''}"
                    for link in content.links
                ]
                return (
                    f"# Links from {content.title}\n\n"
                    f"{content.description}\n\n"
                    f"{chr(10).join(links_text)}"
                )
            else:
                links_text = [
                    f"• {link['text']}: {link['url']}"
                    f"{' (External)' if link['is_external'] else ''}"
                    for link in content.links
                ]
                return (
                    f"Links from: {content.title}\n\n"
                    f"{content.description}\n\n"
                    f"{chr(10).join(links_text)}"
                )
        
        else:  # structured
            if output_format == "markdown":
                # Build structured markdown output
                sections = [
                    f"# {content.title}",
                    f"\n{content.description}\n" if content.description else "",
                    "## Content\n",
                    content.main_content,
                    "\n## Metadata\n",
                    f"- Author: {content.metadata['author'] or 'Unknown'}",
                    f"- Published: {content.metadata['published_date'] or 'Unknown'}",
                    f"- Word Count: {content.metadata['word_count']}",
                    f"- Language: {content.metadata['language'] or 'Unknown'}",
                    f"- Tags: {', '.join(content.metadata['tags']) or 'None'}",
                    "\n## Important Links\n",
                    *[f"- [{link['text']}]({link['url']})" for link in content.links[:5]]
                ]
                return "\n".join(sections)
            else:
                # Build plain text structured output
                sections = [
                    f"Title: {content.title}",
                    f"\nDescription: {content.description}\n" if content.description else "",
                    "Content:",
                    content.main_content,
                    "\nMetadata:",
                    f"• Author: {content.metadata['author'] or 'Unknown'}",
                    f"• Published: {content.metadata['published_date'] or 'Unknown'}",
                    f"• Word Count: {content.metadata['word_count']}",
                    f"• Language: {content.metadata['language'] or 'Unknown'}",
                    f"• Tags: {', '.join(content.metadata['tags']) or 'None'}",
                    "\nImportant Links:",
                    *[f"• {link['text']}: {link['url']}" for link in content.links[:5]]
                ]
                return "\n".join(sections)

    def _generate_suggestions(self, url: str, content: ProcessedContent) -> List[str]:
        """Generate contextual command suggestions."""
        suggestions = [
            f'notes create "Summary of {content.title}"',
            f'web open {url} --extract=links --preview=full',
        ]
        
        # Add search suggestions based on content
        if content.metadata['tags']:
            suggestions.append(
                f'search query "{content.metadata["tags"][0]}"'
            )
        
        # Suggest exploring related links
        for link in content.links[:2]:
            if link['is_external']:
                suggestions.append(f'web open {link["url"]}')
        
        return suggestions