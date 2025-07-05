"""
Custom exceptions for Discord bot operations.
"""


class ContextWindowError(Exception):
    """Base class for context window related errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


class NoContextWindowError(ContextWindowError):
    """Raised when no context window exists for a channel."""
    def __init__(self, message: str = "No context window available. Please fetch history first.", status_code: int = 404):
        super().__init__(message, status_code)


class ContextWindowExpiredError(ContextWindowError):
    """Raised when a context window has expired."""
    def __init__(self, message: str = "Context window has expired. Please refresh history.", status_code: int = 400):
        super().__init__(message, status_code)


class InvalidWindowTimestampError(ContextWindowError):
    """Raised when a context window has an invalid timestamp."""
    def __init__(self, message: str = "Corrupted context window timestamp. Window cleared. Please refresh history.", status_code: int = 500):
        super().__init__(message, status_code)


class ReferenceNotInWindowError(ContextWindowError):
    """Raised when a #N reference is not found in an active window."""
    def __init__(self, message: str = "Reference not found in the current context window. Please refresh history.", status_code: int = 404):
        super().__init__(message, status_code)


class ReferencedMessageNotFound(ContextWindowError):
    """Raised when the message ID from window is not found in Discord (e.g., deleted)."""
    def __init__(self, message: str = "Referenced message not found. It might have been deleted.", status_code: int = 404):
        super().__init__(message, status_code)


class ReferencedMessageForbidden(ContextWindowError):
    """Raised when bot is forbidden to fetch the message ID from window."""
    def __init__(self, message: str = "Bot lacks permissions to fetch the referenced message.", status_code: int = 403):
        super().__init__(message, status_code)


class InvalidMessageIdFormatInWindow(ContextWindowError):
    """Raised if the message ID in the window is not a valid integer."""
    def __init__(self, message: str = "Invalid message ID format in context window. Please refresh context.", status_code: int = 500):
        super().__init__(message, status_code)


# Additional exceptions for the refactored modules
class MappingError(Exception):
    """Base class for name/ID mapping errors."""
    pass


class UserNotFoundError(MappingError):
    """Raised when a user name cannot be mapped to an ID."""
    def __init__(self, username: str, guild_name: str = None):
        self.username = username
        self.guild_name = guild_name
        msg = f"User '{username}' not found"
        if guild_name:
            msg += f" in guild '{guild_name}'"
        super().__init__(msg)


class ChannelNotFoundError(MappingError):
    """Raised when a channel path cannot be mapped to an ID."""
    def __init__(self, channel_path: str):
        self.channel_path = channel_path
        super().__init__(f"Channel '{channel_path}' not found")


class EmojiNotFoundError(MappingError):
    """Raised when an emoji name cannot be mapped to an ID."""
    def __init__(self, emoji_name: str, guild_name: str = None):
        self.emoji_name = emoji_name
        self.guild_name = guild_name
        msg = f"Emoji '{emoji_name}' not found"
        if guild_name:
            msg += f" in guild '{guild_name}'"
        super().__init__(msg)


class CacheError(Exception):
    """Base class for cache-related errors."""
    pass


class CacheTimeoutError(CacheError):
    """Raised when cache operations timeout."""
    pass