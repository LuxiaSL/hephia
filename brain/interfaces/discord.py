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

from brain.prompts.manager import PromptManager
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
            prompt_manager: PromptManager
        ):
        super().__init__("discord", state_bridge, cognitive_bridge, notification_manager, prompt_manager)
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
    
        formatted_notifications = []
        for notif in notifications:
            notification_data = {
                "type": notif.content.get('update_type', 'interaction'),
                "channel": {
                    "name": notif.content.get('channel_name', 'Unknown'),
                    "id": notif.content.get('channel_id', 'Unknown')
                },
                "content": notif.content,
                "message_id": notif.content.get('message_id'),
                "message": notif.content.get('message', ''),
                "response": notif.content.get('response', ''),
                "author": notif.content.get('author', 'Unknown')
            }
            
            # Use prompt manager to format notification
            template_name = (
                "discord_channel" if notification_data["type"] == "channel_activity"
                else "discord_interaction"
            )
            
            formatted = await self.prompt_manager.build_prompt(
                template_name=template_name,
                data=notification_data,
                format_name="notifications"
            )
            formatted_notifications.append(formatted)
        
        return "\n".join(formatted_notifications[-5:])  # Keep last 5 notifications
        
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
        return await self.prompt_manager.build_prompt(
            template_name="base",
            data={
                "state": state,
                "memories": memories
            },
            format_name="cognitive"
        )
    
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
        interaction_data = {
            "channel": metadata.get('channel', {}).get('name', 'Unknown'),
            "guild": metadata.get('channel', {}).get('guild_name', 'DM'),
            "author": metadata.get('message', {}).get('author', 'Unknown'),
            "response": content,
            "history": metadata.get('history', [])
        }

        return await self.prompt_manager.build_prompt(
            template_name="discord",
            data={
                "state": state,
                "interaction": interaction_data
            },
            format_name="memory"
        )

    async def get_relevant_memories(self) -> List[Dict[str, Any]]:
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
        """Format social prompt using prompt system."""
        interaction_data = {
            "channel": {
                "type": "server" if channel_data.get('guild_name') else "DM",
                "name": channel_data.get('name', 'Unknown'),
                "guild": channel_data.get('guild_name', '')
            },
            "author": author,
            "content": message_content,
            "history": history
        }

        return await self.prompt_manager.build_prompt(
            template_name="discord",
            data={
                "base_prompt": Config.DISCORD_SYSTEM_PROMPT,
                "interaction": interaction_data,
                "context": context
            },
            format_name="interface"
        )

    async def _get_social_response(self, prompt: str) -> str:
        """Get LLM response for social interaction."""
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        system_message = await self.prompt_manager.build_prompt(
            template_name="discord",
            data={},
            format_name="system"
        )
        
        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )