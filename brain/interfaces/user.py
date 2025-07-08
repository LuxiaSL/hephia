"""
brain/interfaces/user.py

Implements direct user conversation handling through the User interface.
Maintains cognitive continuity for direct user interactions while managing
proper context and memory formation.
"""

from typing import Dict, List, Any, Optional

from brain.interfaces.base import CognitiveInterface
from brain.cognition.notification import Notification, NotificationManager
from brain.cognition.memory.significance import MemoryData, SourceType
from brain.prompting.loader import get_prompt
from brain.utils.tracer import brain_trace
from config import Config
from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from loggers import BrainLogger
from api_clients import APIManager
from event_dispatcher import global_event_dispatcher, Event

class UserInterface(CognitiveInterface):
    def __init__(
        self,
        api_manager: APIManager,
        state_bridge: StateBridge,
        cognitive_bridge: CognitiveBridge,
        notification_manager: NotificationManager
    ):
        super().__init__("user", state_bridge, cognitive_bridge, notification_manager)
        self.api = api_manager

    @brain_trace
    async def process_interaction(self, content: Dict[str, Any]) -> str:
        """
        Process a user conversation interaction.
        
        Args:
            content: Dictionary containing conversation messages array
                    in OpenAI format (role/content pairs)
        """
        try:
            # Get cognitive context
            brain_trace.interaction.context("Getting cognitive context")
            # Get cognitive context including state and recent activity
            context_parts = {}
            async for key, value in self.get_cognitive_context():
                context_parts[key] = value
            
            # Build final context string from parts
            formatted_context = context_parts.get('formatted_context', 'No context available')
            other_updates = context_parts.get('updates', '')
            context = f"{formatted_context}\n\nRecent Updates:\n{other_updates}"
            BrainLogger.info(f"[{self.interface_id}] Received cognitive context: {context}")
            brain_trace.interaction.context.complete(context=context)
            
            # Format conversation for processing
            result = await self._process_conversation(
                conversation=content.get('messages', []),
                context=context
            )
            
            # Create notification for other interfaces
            notification = await self.create_notification({
                "response": result,
                "conversation": content.get('messages', [])[-3:] if content.get('messages') else [],
                "interaction_type": "conversation",
                "message_count": len(content.get('messages', []))
            })

            await self.announce_cognitive_context([result, content.get('messages', [])], notification)
            
            await self._dispatch_memory_check(result, content.get('messages', []), context)

            return result
            
        except Exception as e:
            BrainLogger.error(f"Error processing user interaction: {e}")
            return "I apologize, but I'm having trouble processing at the moment."
        
    async def _dispatch_memory_check(
        self,
        response: str,
        conversation: List[Dict[str, str]],
        cognitive_updates: str
    ) -> None:
        """Dispatch memory check for direct user interaction."""
        recent_messages = conversation[-3:] if len(conversation) > 3 else conversation
        
        try:
            memory_data = MemoryData(
                interface_id=self.interface_id,
                content=response,
                context=await self.state_bridge.get_api_context(),
                source_type=SourceType.DIRECT_CHAT,
                metadata={
                    'conversation': {
                        'recent_messages': recent_messages,
                        'total_messages': len(conversation),
                        'last_user_message': next(
                            (msg['content'] for msg in reversed(conversation) 
                             if msg['role'] == 'user'),
                            None
                        )
                    },
                    'interaction_depth': len(conversation),
                    'has_multi_turn': len(conversation) > 2,
                    'cognitive_updates': cognitive_updates,
                }
            )
        except Exception as e:
            BrainLogger.error(f"Error creating direct chat memory data: {e}", exc_info=True)
            return

        BrainLogger.info(f"Memory data for dispatch (Direct Chat): {memory_data}")

        try:
            final_event_data = {
                "event_type": "direct_chat",
                "content": memory_data.content,
                "event_data": memory_data.to_event_data()
            }
            global_event_dispatcher.dispatch_event(
                Event(f"cognitive:{self.interface_id}:memory_check", final_event_data)
            )
            BrainLogger.info("Direct chat memory event dispatched successfully.")
        except Exception as e:
            BrainLogger.error(f"Error dispatching direct chat memory event: {e}", exc_info=True)

    async def format_memory_context(
        self,
        content: Any,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format memory context for user conversations.
        
        Args:
            content: Response content
            state: Current cognitive state
            metadata: Contains conversation messages and context
        """
        conversation = metadata.get('conversation', [])
        
        # Format recent conversation context
        conversation_context = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in conversation[-5:]  # Last 5 messages
        ]) if conversation else "No conversation context"

        return get_prompt(
            "interfaces.user.memory.template",
            model=Config.get_cognitive_model(),
            vars={
                "conversation_context": conversation_context,
                "content": content,
                'context': metadata.get('cognitive_updates', ''),
            }
        )
    
    async def get_fallback_memory(self, memory_data: MemoryData) -> Optional[str]:
        """
        Generate a fallback memory for user direct conversations when primary memory generation fails.
        
        Args:
            memory_data: Memory data object containing interaction context and content
        """
        try:
            # Extract core components from memory data
            conversation_data = memory_data.metadata.get('conversation', {})
            last_user_message = conversation_data.get('last_user_message', 'Unknown message')

            # Create simple structured fallback memory
            memory_parts = [
                "Direct user conversation:",
                f"User said: {last_user_message}",
                f"I responded: {memory_data.content}"
            ]

            return "\n".join(memory_parts)

        except Exception as e:
            BrainLogger.error(f"Error generating user conversation fallback memory: {e}")
            return None

    async def get_relevant_memories(self, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to user conversations.
        Uses current conversation context for targeted memory retrieval.
        """
        # Default query if no metadata available
        query = "user conversation interaction discussion"
        
        if metadata:
            # Extract conversation messages for context
            conversation = metadata.get('conversation', [])
            
            if conversation:
                # Get recent messages for context (last 3-5 messages)
                recent_messages = conversation[-5:]
                
                # Build context from recent conversation
                context_parts = []
                for msg in recent_messages:
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    if content:
                        context_parts.append(f"{role}: {content}")
                
                # Use conversation context as query if available
                if context_parts:
                    query = " ".join(context_parts)
        
        return await self.cognitive_bridge.retrieve_memories(
            query=query,
            limit=3
        )

    async def _process_conversation(
        self,
        conversation: List[Dict[str, str]],
        context: str
    ) -> str:
        """Process user conversation with context."""
        try:
            model_name = Config.get_cognitive_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            # Create conversation with context
            messages = [
                {
                    "role": "system",
                    "content": get_prompt(
                        "interfaces.user.interaction.system",
                        model=model_name
                    )
                },
                {
                    "role": "system",
                    "content": f"Current Context:\n{context}"
                }
            ]
            
            # Add conversation messages
            messages.extend(conversation)
            
            return await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=messages,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )
            
        except Exception as e:
            BrainLogger.error(f"Conversation processing failed: {e}")
            raise

    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """Generate summary of recent user interactions."""
        formatted_interactions = []
        
        for notif in notifications:
            if isinstance(notif.content, dict):
                # Extract conversation and response
                recent_msgs = notif.content.get('conversation', [])
                response = notif.content.get('response', '')
                
                # Get last user message if available
                last_user_msg = next(
                    (msg['content'] for msg in reversed(recent_msgs) 
                     if msg['role'] == 'user'),
                    'No message'
                )
                
                formatted_interactions.append(
                    f"- User said: {last_user_msg}\n  "
                    f"Response: {response[:250]}..."
                )
        
        # Take last 5 interactions
        interactions_text = "\n".join(formatted_interactions[-5:]) if formatted_interactions else "No recent interactions"
        
        return f"""Recent User Interactions:
{interactions_text}"""