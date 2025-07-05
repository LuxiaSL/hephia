"""
Data models for Discord bot operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import discord


@dataclass
class EnhancedMessage:
    """
    Enhanced message model with bidirectional format support.
    
    This model stores both LLM-friendly and Discord-native formats,
    allowing clean abstraction while maintaining full Discord functionality.
    """
    id: str
    author: str  # LLM-friendly name (e.g., "john")
    discord_author_id: str  # Discord user ID for internal use
    content: str  # LLM-friendly format (@john, :emoji:)
    raw_content: str  # Original Discord format (<@123>, <:emoji:456>)
    timestamp: datetime
    channel_path: str  # LLM-friendly path (e.g., "MyServer:general")
    channel_id: str  # Discord channel ID for internal use
    reference: Optional[str] = None  # #N reference if in context window
    reply_to: Optional[str] = None  # Message ID this is replying to
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    embeds: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_llm_format(self) -> Dict[str, Any]:
        """Convert to LLM-friendly format for API responses."""
        result = {
            "id": self.id,
            "author": self.author,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "channel_path": self.channel_path
        }
        
        if self.reference:
            result["reference"] = self.reference
            
        if self.reply_to:
            result["reply_to"] = self.reply_to
            
        if self.attachments:
            result["attachments"] = self.attachments
            
        return result
    
    def to_discord_format(self) -> Dict[str, Any]:
        """Convert to Discord API format for sending."""
        return {
            "content": self.raw_content,
            "author_id": self.discord_author_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ContextWindow:
    """
    Context window with expiration and message references.
    
    Manages the #N reference system that allows LLMs to reference
    messages without dealing with Discord message IDs.
    """
    channel_id: str
    channel_path: str  # LLM-friendly path
    messages: Dict[str, str] = field(default_factory=dict)  # #N -> message_id
    timestamp: datetime = field(default_factory=datetime.now)
    expiry_minutes: int = 5
    
    @property
    def expires_at(self) -> datetime:
        """Calculate expiration time."""
        from datetime import timedelta
        return self.timestamp + timedelta(minutes=self.expiry_minutes)
    
    def is_expired(self) -> bool:
        """Check if context window has expired."""
        return datetime.now() > self.expires_at
        
    def resolve_reference(self, reference: str) -> Optional[str]:
        """Resolve a #N reference to a message ID."""
        return self.messages.get(reference)
    
    def add_message_reference(self, reference: str, message_id: str) -> None:
        """Add a message reference to the window."""
        self.messages[reference] = message_id
    
    def get_all_references(self) -> Dict[str, str]:
        """Get all message references in the window."""
        return self.messages.copy()


@dataclass
class UserMapping:
    """
    User mapping data for bidirectional name/ID conversion.
    """
    discord_id: str
    username: str  # Discord username
    display_name: str  # Display name in guild
    global_name: Optional[str] = None  # New Discord global name
    guild_id: Optional[str] = None  # Which guild this mapping is for
    last_seen: datetime = field(default_factory=datetime.now)
    
    @property
    def llm_name(self) -> str:
        """Get the name that should be shown to LLM."""
        # Prefer display name, fall back to username
        return self.display_name or self.username
    
    @property
    def discord_mention(self) -> str:
        """Get Discord mention format."""
        return f"<@{self.discord_id}>"


@dataclass
class ChannelMapping:
    """
    Channel mapping data for path/ID conversion.
    """
    discord_id: str
    guild_id: str
    channel_name: str
    guild_name: str
    channel_type: str  # 'text', 'voice', 'thread', etc.
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def llm_path(self) -> str:
        """Get LLM-friendly path."""
        return f"{self.guild_name}:{self.channel_name}"
    
    @property
    def path_variations(self) -> List[str]:
        """Get all possible path variations for matching."""
        return [
            self.llm_path,
            self.llm_path.lower(),
            f"{self.guild_name.lower()}:{self.channel_name.lower()}"
        ]


@dataclass
class EmojiMapping:
    """
    Custom emoji mapping for bidirectional conversion.
    """
    discord_id: str
    name: str
    guild_id: str
    animated: bool = False
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def llm_format(self) -> str:
        """Get LLM-friendly format."""
        return f":{self.name}:"
    
    @property
    def discord_format(self) -> str:
        """Get Discord format."""
        prefix = "a" if self.animated else ""
        return f"<{prefix}:{self.name}:{self.discord_id}>"


@dataclass
class MessageCacheStats:
    """
    Statistics for message cache performance monitoring.
    """
    total_messages: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    @property
    def api_efficiency(self) -> float:
        """Calculate API call efficiency (messages per API call)."""
        return self.total_messages / self.api_calls if self.api_calls > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "total_messages": self.total_messages,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "last_reset": self.last_reset.isoformat(),
            "hit_rate": self.hit_rate,
            "api_efficiency": self.api_efficiency
        }


@dataclass
class BotConfig:
    """
    Configuration settings for the Discord bot.
    """
    max_message_cache_size: int = 1000
    context_window_expiry_minutes: int = 5
    max_concurrent_requests: int = 5
    enable_debug_logging: bool = False
    message_chunk_size: int = 1750  # Discord's practical limit
    
    # Engagement settings
    high_message_threshold: int = 50
    random_engagement_rate: float = 0.75
    
    # Cache settings
    cache_cleanup_interval: int = 300  # 5 minutes
    max_cache_age_hours: int = 24
    
    # Rate limiting
    requests_per_minute: int = 60
    burst_size: int = 10


@dataclass
class BotStatistics:
    """
    Current status of the Discord bot.
    """
    is_ready: bool = False
    connected_guilds: int = 0
    total_channels: int = 0
    cached_messages: int = 0
    active_context_windows: int = 0
    uptime_start: datetime = field(default_factory=datetime.now)
    last_message_time: Optional[datetime] = None
    
    @property
    def uptime_seconds(self) -> float:
        """Get uptime in seconds."""
        return (datetime.now() - self.uptime_start).total_seconds()
    
    @property
    def uptime_formatted(self) -> str:
        """Get formatted uptime string."""
        total_seconds = int(self.uptime_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "is_ready": self.is_ready,
            "connected_guilds": self.connected_guilds,
            "total_channels": self.total_channels,
            "cached_messages": self.cached_messages,
            "active_context_windows": self.active_context_windows,
            "uptime_start": self.uptime_start.isoformat(),
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "uptime_seconds": self.uptime_seconds,
            "uptime_formatted": self.uptime_formatted
        }


@dataclass
class MessageFormatResult:
    """
    Result of message format conversion.
    """
    success: bool
    llm_content: str
    discord_content: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False