"""
brain/interfaces/discord_interface.py

Implements the Discord interface for social interactions through Discord.
Handles message processing, memory formation, and cognitive continuity
while maintaining natural social engagement patterns.
"""

from typing import Dict, List, Any, Optional

from brain.environments.terminal_formatter import TerminalFormatter
from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from brain.interfaces.base import CognitiveInterface
from brain.cognition.notification import Notification, NotificationManager

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
            context = f"{formatted_context}\n\nRecent Updates:\n{other_updates}"
            
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
            response = await self._get_social_response(prompt)
            
            # Create notification for other interfaces
            notification = await self.create_notification({
                "response": response,
                "message_id": message.get('id'),
                "message": message_content,
                "author": author,
                "channel": channel,
                "guild": channel_data.get('guild_name'),
                "channel_type": "server" if channel_data.get('guild_name') else "DM",
                "timestamp": message.get('timestamp')
            })

            # Tell everyone about the update
            await self.announce_cognitive_context([response, content], notification)

            # Memory processing
            await self._dispatch_memory_check(response, content)
            
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
            return "I apologize, but I'm having trouble processing right now."

    # Keep existing methods
    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """Generate a summary of this interface's provided notifications."""
        BrainLogger.debug(f"notifications: {notifications}")
        formatted = []
        for notif in notifications:
            BrainLogger.debug(f"notif: {notif}, content: {notif.content}")
            
            if notif.content.get('update_type') == 'channel_activity':
                # Handle channel activity updates
                message = (
                    f"New messages detected in Discord channel {notif.content.get('channel_name', 'Unknown')}"
                    f"(ID: {notif.content.get('channel_id', 'Unknown')})"
                )
                formatted.append(message)
            else:
                # Handle conversation notifications
                content = notif.content
                summary_text = (
                    f"Discord update: Replied to {content.get('author', 'Unknown')} in channel {content.get('channel', 'Unknown')}\n"
                    f"- Message ID: {content.get('message_id', 'Unknown')}\n"
                    f"- User said: {content.get('message', '')[:250]}{'...' if len(content.get('message', '')) > 250 else ''}\n"
                    f"- My response: {content.get('response', '')[:50]}{'...' if len(content.get('response', '')) > 50 else ''}"
                )
                formatted.append(summary_text)
        
        summary = "\n".join(formatted[-5:])  # Last 5 notifications
        return f"""Recent Discord Activity:
    {summary}"""
        
    async def _dispatch_memory_check(
        self,
        response: str,
        message_context: Dict[str, Any]
    ) -> None:
        """Dispatch memory check for Discord interaction."""
        try:
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
                        'name': message_context.get('channel', {}).get('name'),
                        'guild': message_context.get('channel', {}).get('guild_name'),
                        'is_dm': not message_context.get('channel', {}).get('guild_name')
                    },
                    'history': message_context.get('conversation_history', [])[-3:],
                    'mentions_bot': '@hephia' in (
                        message_context.get('current_message', {}).get('content', '').lower()
                    )
                }
            )
        except Exception as e:
            BrainLogger.error(f"Error creating Discord memory data: {e}", exc_info=True)
            return
        
        BrainLogger.info(f"Memory data for dispatch (Discord): {memory_data}")

        try:
            # Wrap the MemoryData output in the expected legacy structure.
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

    async def format_cognitive_context(
        self,
        state: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """Format cognitive context for social interactions."""
        # Format state information
        return TerminalFormatter.format_context_summary(state, memories)
    
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
        channel = message_data.get('channel', {}).get('name', 'Unknown')
        guild = message_data.get('channel', {}).get('guild_name', 'DM')
        
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
                        f"[{date} {time}] {msg.get('author', 'Unknown')}: {msg.get('content', '')}"
                    )
                except (KeyError, IndexError, AttributeError):
                    # Fallback for malformed entries
                    history_entries.append(f"{msg.get('author', 'Unknown')}: {msg.get('content', '')}")
            
            history_text = "\n".join(history_entries)

        return f"""Form a memory of this Discord interaction:

Context:
Channel: #{channel} in {guild}
Conversation with: {author}

Recent History:
{history_text}

My Response: {content}

Create a concise first-person memory snippet that captures:
1. The social dynamics and emotional context
2. Any relationship developments or insights
3. Key points of the conversation
4. Thoughts and reactions
"""

    async def get_relevant_memories(self, metadata: Optional[str]) -> List[Dict[str, Any]]:
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
        
        return f"""Process this Discord interaction from your perspective:

Environment: Discord {channel_type} (#{channel_name})
From: {author}
Message: {message_content}

Recent Conversation:
{history_text}

My Current Context:
{context}"""

    async def _get_social_response(self, prompt: str) -> str:
        """Get LLM response for social interaction."""
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        system_prompt = Config.DISCORD_SYSTEM_PROMPT
        
        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )