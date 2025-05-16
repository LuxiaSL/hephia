"""
brain/interfaces/discord_interface.py

Implements the Discord interface for social interactions through Discord.
Handles message processing, memory formation, and cognitive continuity
while maintaining natural social engagement patterns.
"""

import asyncio
from typing import Dict, List, Any, Optional

from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from brain.interfaces.base import CognitiveInterface
from brain.cognition.notification import Notification, NotificationManager
from brain.prompting.loader import get_prompt
from brain.cognition.memory.significance import MemoryData, SourceType
from brain.utils.tracer import brain_trace
from config import Config
from loggers import BrainLogger
from api_clients import APIManager
from event_dispatcher import global_event_dispatcher, Event

class DiscordInterface(CognitiveInterface):
    def __init__(
            self,
            api_manager: APIManager,
            state_bridge: StateBridge,
            cognitive_bridge: CognitiveBridge,
            notification_manager: NotificationManager,
        ):
        super().__init__("discord", state_bridge, cognitive_bridge, notification_manager)
        self.api = api_manager
        
    @brain_trace
    async def process_interaction(self, content: Dict[str, Any]) -> str:
        """Process a Discord message interaction."""
        try:
            brain_trace.interaction.start(
                metadata={
                    "interface": "discord",
                    "message_id": content.get('current_message', {}).get('id'),
                    "channel": content.get('channel', {}).get('name'),
                    "guild": content.get('channel', {}).get('guild_name')
                }
            )

            # Extract message details
            message = content.get('current_message', {})
            message_content = message.get('content', '')
            author = message.get('author', 'Unknown')
            channel_data = content.get('channel', {})
            channel = channel_data.get('name', 'Unknown')
            
            # Get cognitive context including state and recent activity
            context_parts = {}
            async for key, value in self.get_cognitive_context():
                context_parts[key] = value
            
            # Build final context string from parts
            formatted_context = context_parts.get('formatted_context', 'No context available')
            other_updates = context_parts.get('updates', '')
            context = f"{formatted_context}\n###\nRecent Updates:\n{other_updates}"
            
            # Log the entire context for debugging
            BrainLogger.info(f"[{self.interface_id}] Received cognitive context: {context}")
            
            # Get LLM response
            prompt = await self._format_social_prompt(
                message_content,
                author,
                channel_data,
                content.get('conversation_history', []),
                context
            )

            max_retries = 3
            response = None

            if Config.get_discord_reply_on_tag():
                for attempt in range(max_retries):
                    response = await self._get_social_response(prompt)
                    if response is not None:
                        break
                    # Wait 2^attempt seconds before retrying (1s, 2s, 4s)
                    await asyncio.sleep(2 ** attempt)
                
                # If all retries failed, return empty string
                if response is None:
                    BrainLogger.error("Failed to get social response after 3 attempts")
            
            # Create notification for other interfaces
            notification = await self.create_notification({
                "response": response,
                "message_id": message.get('id'),  # Keep for internal use but will hide from display
                "message": message_content,
                "author": author,
                "channel": channel,
                "guild": channel_data.get('guild_name'),
                "path": f"{channel_data.get('guild_name', 'Unknown')}:{channel}" if channel_data.get('guild_name') else channel,
                "channel_type": "server" if channel_data.get('guild_name') else "DM",
                "timestamp": message.get('timestamp'),
                "metadata": content.get('metadata', {}),
            })

            # Tell everyone about the update
            await self.announce_cognitive_context([response, content], notification)

            # Memory processing
            await self._dispatch_memory_check(response, content, context)
            
            return response
            
        except Exception as e:
            brain_trace.error.process_interaction(
                error=e,
                context={
                    "message_content": message_content,
                    "author": author,
                    "channel": channel,
                    "error_type": type(e).__name__,
                    "error_details": str(e)
                }
            )
            BrainLogger.error(f"Error processing Discord interaction: {e}", exc_info=True)
            return ""

    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """Generate a summary of this interface's provided notifications."""
        BrainLogger.debug(f"notifications: {notifications}")
        formatted = []
        for notif in notifications:
            BrainLogger.debug(f"notif: {notif}, content: {notif.content}")
            
            if notif.content.get('update_type') == 'channel_activity':
                # Handle channel activity updates using path format
                channel_name = notif.content.get('channel_name', 'Unknown')
                guild_name = notif.content.get('guild_name', 'Unknown')
                path = f"{guild_name}:{channel_name}" if guild_name else channel_name
                message = f"New messages detected in Discord channel {path}"
                formatted.append(message)
            else:
                # Handle conversation notifications with path format
                content = notif.content
                # Use path directly if available
                path = content.get('path')
                
                # If path not available, construct it from guild and channel
                if not path:
                    channel = content.get('channel', 'Unknown')
                    guild = content.get('guild')
                    path = f"{guild}:{channel}" if guild else channel

                author = content.get('author', 'Unknown')
                
                summary_text = (
                    f"Discord update: Replied to {author} in {path}\n"
                    f"{author}: {content.get('message', '')}\n"
                    f"I responded: {content.get('response', '')[:150]}{'...' if len(content.get('response', '')) > 150 else ''}" if content.get("response") != None else None
                )
                formatted.append(summary_text)
        
        if not formatted:
            summary = "No recent notifications."
        else:
            summary = "\n###".join(formatted)

        return f"""Recent Discord Activity:
    {summary}"""
        
    async def _dispatch_memory_check(
        self,
        response: str,
        message_context: Dict[str, Any],
        cognitive_updates: str
    ) -> None:
        """Dispatch memory check for Discord interaction."""
        try:
            # Extract channel and guild information
            channel_name = message_context.get('channel', {}).get('name', 'Unknown')
            guild_name = message_context.get('channel', {}).get('guild_name')
            
            # Create path format for channel reference
            channel_path = f"{guild_name}:{channel_name}" if guild_name else channel_name
            
            memory_data = MemoryData(
                interface_id=self.interface_id,
                content=response,
                context=await self.state_bridge.get_api_context(),
                source_type=SourceType.DISCORD,
                metadata={
                    'message': {
                        'content': message_context.get('current_message', {}).get('content'),
                        'author': message_context.get('current_message', {}).get('author'),
                        'timestamp': message_context.get('current_message', {}).get('timestamp')
                    },
                    'channel': {
                        'name': channel_name,
                        'guild': guild_name,
                        'path': channel_path,
                        'is_dm': not guild_name
                    },
                    'history': message_context.get('conversation_history', [])[-3:],
                    'mentions_bot': '@hephia' in (
                        message_context.get('current_message', {}).get('content', '').lower()
                    ),
                    'cognitive_updates': cognitive_updates
                }
            )
        except Exception as e:
            BrainLogger.error(f"Error creating Discord memory data: {e}", exc_info=True)
            return
        
        BrainLogger.info(f"Memory data for dispatch (Discord): {memory_data}")

        try:
            final_event_data = {
                "event_type": "discord",
                "content": memory_data.content,
                "event_data": memory_data.to_event_data()
            }
            global_event_dispatcher.dispatch_event(
                Event(f"cognitive:{self.interface_id}:memory_check", final_event_data)
            )
            BrainLogger.info("Discord memory event dispatched successfully.")
        except Exception as e:
            BrainLogger.error(f"Error dispatching Discord memory event: {e}", exc_info=True)
    
    async def format_memory_context(
        self,
        content: Any,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format social interaction context for memory formation.
        
        Args:
            content: Response content
            state: Current cognitive state
            metadata: Contains message and channel information
        """
        message_data = metadata.get('message', {})
        author = message_data.get('author', 'Unknown')
        
        # Use path directly if available in channel metadata
        channel_data = metadata.get('channel', {})
        channel_path = channel_data.get('path')
        
        # If path not provided, construct it from guild and channel
        if not channel_path:
            channel_name = channel_data.get('name', 'Unknown')
            guild_name = channel_data.get('guild')
            channel_path = f"{guild_name}:{channel_name}" if guild_name and guild_name != 'DM' else channel_name
        
        history = metadata.get('history', [])
        history_text = ""
        if history:
            history_entries = []
            for msg in history:
                try:
                    # Use proper timestamp field from message
                    timestamp = msg.get('timestamp', '').split('.')[0]  # Remove microseconds
                    date = timestamp.split('T')[0]
                    time = timestamp.split('T')[1][:8]
                    history_entries.append(
                        f"[{date} {time}] <{msg.get('author', 'Unknown')}>: {msg.get('content', '')}"
                    )
                except (KeyError, IndexError, AttributeError):
                    # Fallback for malformed entries
                    history_entries.append(f"<{msg.get('author', 'Unknown')}>: {msg.get('content', '')}")
            
            history_text = "\n".join(history_entries)

        return get_prompt(
            "interfaces.discord.memory.template",
            model=Config.get_cognitive_model(),
            vars={
                "channel_path": channel_path,
                "author": author,
                "history_text": history_text,
                "content": content,
                'context': metadata.get('cognitive_updates', ''),
            }
        )
    
    async def get_fallback_memory(self, memory_data: MemoryData) -> Optional[str]:
        """
        Generate a fallback memory for Discord interactions when primary memory generation fails.
        
        Args:
            memory_data: Memory data object containing interaction context and content
        """
        try:
            # Extract core components from memory data
            message_data = memory_data.metadata.get('message', {})
            channel_data = memory_data.metadata.get('channel', {})
            history = memory_data.metadata.get('history', [])[-5:]  
            
            # Get channel path
            channel_path = channel_data.get('path')
            if not channel_path:
                channel_name = channel_data.get('name', 'Unknown')
                guild_name = channel_data.get('guild')
                channel_path = f"{guild_name}:{channel_name}" if guild_name and guild_name != 'DM' else channel_name

            # Format recent history concisely
            history_parts = []
            for msg in history:
                author = msg.get('author', 'Unknown')
                content = msg.get('content', '')[:150]  # Trim long messages
                history_parts.append(f"<{author}>: {content}...")

            # Create structured fallback memory
            memory_parts = [
                f"Discord interaction in {channel_path}",
                f"Conversation with {message_data.get('author', 'Unknown')}:",
                *history_parts,
                f"My response: {memory_data.content}"
            ]

            return "\n".join(memory_parts)

        except Exception as e:
            BrainLogger.error(f"Error generating Discord fallback memory: {e}")
            return None

    async def get_relevant_memories(self, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to current social context.
        Focuses on interaction patterns and relationship context.
        """
        query = f"social interaction discord conversation relationship"
        return await self.cognitive_bridge.retrieve_memories(
            query=query,
            limit=3
        )

    async def _format_social_prompt(
        self,
        message_content: str,
        author: str,
        channel_data: Dict[str, Any],
        history: List[Dict[str, Any]],
        context: str
    ) -> str:
        """Format prompt for social interaction."""
        # Format conversation history
        history_text = ""
        if history:
            history_entries = []
            for msg in history:
                try:
                    # Use proper timestamp field from message
                    timestamp = msg.get('timestamp', '').split('.')[0]  # Remove microseconds
                    date = timestamp.split('T')[0]
                    time = timestamp.split('T')[1][:8]
                    history_entries.append(
                        f"[{date} {time}] {msg.get('author', 'Unknown')}: {msg.get('content', '')}"
                    )
                except (KeyError, IndexError, AttributeError):
                    # Fallback for malformed entries
                    history_entries.append(f"{msg.get('author', 'Unknown')}: {msg.get('content', '')}")
            
            history_text = "\n".join(history_entries)
            
        channel_type = "server" if channel_data.get('guild_name') else "DM"
        channel_name = channel_data.get('name', 'Unknown')
        guild_name = channel_data.get('guild_name')
        
        # Create path format for channel
        channel_path = f"{guild_name}:{channel_name}" if guild_name else channel_name
        
        return get_prompt(
            "interfaces.discord.interaction.user",
            model=Config.get_cognitive_model(),
            vars={
                "channel_type": channel_type,
                "channel_path": channel_path,
                "author": author,
                "message_content": message_content,
                "history_text": history_text,
                "context": context
            }
        )

    async def _get_social_response(self, prompt: str) -> str:
        """Get LLM response for social interaction."""
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=[
                {"role": "system", "content": get_prompt(
                    "interfaces.discord.interaction.system",
                    model=model_name
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )