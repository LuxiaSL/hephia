"""
brain/interfaces/exo.py

Implements the ExoProcessor interface for the command-based terminal environment.
Handles command processing, LLM interactions, and maintenance of cognitive continuity
for the core command loop while integrating with the broader cognitive architecture.
"""
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional

from brain.cognition.notification import Notification, NotificationManager
from brain.cognition.memory.significance import MemoryData, SourceType
from brain.commands.model import CommandResult, ParsedCommand, GlobalCommands
from brain.core.command_handler import CommandHandler
from brain.environments.terminal_formatter import TerminalFormatter
from brain.interfaces.base import CognitiveInterface
from brain.utils.tracer import brain_trace
from brain.prompting.loader import get_prompt
from brain.interfaces.exo_utils.conversation_state import ConversationState
from brain.interfaces.exo_utils.hud.construct import HudConstructor

from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge

from config import Config
from loggers import BrainLogger
from api_clients import APIManager
from event_dispatcher import global_event_dispatcher, Event

class ExoProcessorInterface(CognitiveInterface):
    def __init__(
        self,
        api_manager: APIManager,
        cognitive_bridge: CognitiveBridge,
        state_bridge: StateBridge,
        command_handler: CommandHandler,
        notification_manager: NotificationManager,
        hud_constructor: HudConstructor
    ):
        super().__init__("exo_processor", state_bridge, cognitive_bridge, notification_manager)
        self.api = api_manager
        self.command_handler = command_handler
        self.hud_constructor = hud_constructor
        self.conversation_state = None
        self.last_successful_turn = None
        self.last_summary_time = None
        self.cached_summary = ""

        self.hud_metadata: Dict[str, Any] = {
            "last_discord_channel_path": None,
            "conversation_state_size": 0,
            "current_goal": None,
            "last_interaction_time": None,
            "current_interaction_time": None,
            "memories_block": None,
            "state_block": None,
        }

        # Timing configuration
        self.min_interval = Config.get_exo_min_interval()
        self.llm_timeout = Config.LLM_TIMEOUT

    async def initialize(self, brain_state = None) -> None:
        """Initialize the exo processor interface and conversation history."""
        if brain_state and len(brain_state) > 2:
            try:
                # Check if the last message is from assistant and remove if so
                if brain_state[-1]["role"] == "assistant":
                    brain_state = brain_state[:-1]
                # Create conversation state from existing messages 
                self.conversation_state = ConversationState.from_message_list(brain_state)

                # reconstruct minimal post-action hud metadata
                self.hud_metadata["conversation_state_size"] = len(self.conversation_state.pairs) * 2
                self.hud_metadata["current_interaction_time"] = datetime.now()
                self.hud_metadata["last_interaction_time"] = datetime.now()
                BrainLogger.info("Restored conversation state from brain state")
                return
            except ValueError as e:
                BrainLogger.error(f"Error restoring brain state: {e}, starting fresh")
        
        # first/default run
        model_name = Config.get_cognitive_model()
        initial_state = await self.state_bridge.get_api_context()
        formatted_state = TerminalFormatter.format_context_summary(initial_state)
        
        self.conversation_state = ConversationState.create_initial(
            system_content=get_prompt(
                'interfaces.exo.turn.system',
                model=model_name
            ),
            user_content=get_prompt(
                'interfaces.exo.welcome.template',
                model=model_name,
                vars={
                    "state_summary": formatted_state
                }
            )
        )
        self.hud_metadata["conversation_state_size"] = len(self.conversation_state.pairs) * 2
        self.hud_metadata["current_interaction_time"] = datetime.now()
        self.hud_metadata["last_interaction_time"] = datetime.now()
        # Reset timing trackers
        self.last_successful_turn = None

    @brain_trace
    async def process_interaction(self, content: Any) -> Any:
        """
        Process one turn of the command-based interaction loop.
        
        Args:
            content: Current turn's input (usually None for ExoProcessor
                    as it maintains its own conversation state)
        """
        try:
            async with asyncio.timeout(self.llm_timeout):
                if not self.conversation_state:
                    # Initialize with a basic state if none exists
                    BrainLogger.error("No or invalid conversation state found, re-initializing")
                    await self.initialize()

                # Get cognitive context
                brain_trace.interaction.context("Getting HUD context")
                context_data = {}
                async for key, value in self.get_cognitive_context():
                    context_data[key] = value
                    #now also has memories/state to be sent into hud metadata
                
                other_updates = context_data.get("updates", [])

                # update HUD metadata with necessary pieces for *pre-turn construction*
                self.hud_metadata["current_interaction_time"] = datetime.now()
                self.hud_metadata["memories_block"] = context_data.get("memories", [])
                self.hud_metadata["state_block"] = context_data.get("state", {})

                context = await self.hud_constructor.build_hud_string(self.hud_metadata, Config.get_cognitive_model())

                BrainLogger.info(f"[{self.interface_id}] Received HUD context: {context}")
                brain_trace.interaction.context.complete(context=context)
                
                # Get LLM response
                brain_trace.interaction.llm("Getting LLM response")
                llm_response = await self._get_llm_response(context)
                
                # Process command
                brain_trace.interaction.command("Processing command")
                command, error = await self.command_handler.preprocess_command(llm_response)

                log_path = "data/logs/exo_conversation.log"

                BrainLogger.debug(f"collected notifications: {other_updates}")

                if error:
                    # cleaning only for when hallucinated context is terribly long
                    cleaned_response = await self.clean_errored_response(llm_response)
                    formatted_error = TerminalFormatter.format_notifications(other_updates, error)
                    self.conversation_state.add_exchange(
                        assistant_content=cleaned_response,
                        user_content=formatted_error,
                        assistant_metadata={"error": True},
                        user_metadata={"error_message": error}
                    )
                    brain_trace.interaction.error(error_msg=error)
                    #log it anyways
                    try:
                        with open(log_path, "w", encoding="utf-8") as f:
                            for message in self.conversation_state.to_message_list():
                                f.write(f"<{message['role']}> {message['content']}\n")
                    except Exception as e:
                        BrainLogger.error(f"Failed to write conversation log: {e}")
                    return error
                
                # Execute command
                brain_trace.interaction.execute("Executing command")
                result = await self.command_handler.execute_command(command)
                self.last_successful_turn = datetime.now()
                
                # Format response
                brain_trace.interaction.format("Formatting response")
                formatted_response = TerminalFormatter.format_command_result(result)
                final_response = TerminalFormatter.format_notifications(other_updates, formatted_response)
                BrainLogger.debug(f"Final Response: {final_response}")
                # Update conversation
                brain_trace.interaction.update("Updating conversation state")

                response_to_return = TerminalFormatter.format_command_cleanly(command)
                
                self.conversation_state.add_exchange(
                    assistant_content=response_to_return,
                    user_content=final_response,
                    assistant_metadata={"command": command if command else None},
                    user_metadata={"result": result if result else None}
                )
                self.conversation_state.trim_to_size(Config.get_exo_max_turns())

                # collecting metadata for HUD
                if result.success:
                    if command.environment == "discord" and result.data and "discord_channel_path" in result.data:
                        new_path = result.data["discord_channel_path"]
                        if self.hud_metadata.get("last_discord_channel_path") != new_path:
                            BrainLogger.info(f"HUD: Discord channel changed to: {new_path}")
                            self.hud_metadata["last_discord_channel_path"] = new_path
                
                self.hud_metadata["conversation_state_size"] = len(self.conversation_state.pairs)
                self.hud_metadata["last_interaction_time"] = self.hud_metadata["current_interaction_time"]

                # Log conversation history to file
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        for message in self.conversation_state.to_message_list():
                            f.write(f"<{message['role']}> {message['content']}\n")
                except Exception as e:
                    BrainLogger.error(f"Failed to write conversation log: {e}")
                
                # Create notification for other interfaces
                brain_trace.interaction.notify("Final updates")
                # Create notification with command results
                notification = await self.create_notification({
                    "response": llm_response,
                    "command": command.raw_input,
                    "result": result.message,
                    "success": result.success,
                    "data": result.data or {},
                    "environment": command.environment,
                    "action": command.action,
                    "state_changes": result.state_changes or {}
                })

                await self.announce_cognitive_context(self.conversation_state.to_message_list(), notification)
                await self._dispatch_memory_check(formatted_response, command, result, other_updates)
                
                return formatted_response
                
        except asyncio.TimeoutError as e:
            brain_trace.error(
                error=e,
                context={"timeout": self.llm_timeout}
            )
            return None
        except Exception as e:
            brain_trace.error(
                error=e,
                context={
                    "conversation_length": len(self.conversation_state.to_message_list()),
                    "last_successful": self.last_successful_turn.isoformat() if self.last_successful_turn else None
                }
            )
            return None
        
    async def _dispatch_memory_check(self, response: str, command: ParsedCommand, result: CommandResult, cognitive_updates: str) -> None:
        BrainLogger.info("Starting memory check dispatch")
        context = await self.state_bridge.get_api_context()
        BrainLogger.info(f"Context for memory check: {context}")
        
        try:
            # Create MemoryData with the actual ParsedCommand and CommandResult instances.
            memory_data = MemoryData(
                interface_id=self.interface_id,
                content=response,
                context=context,
                source_type=SourceType.COMMAND,
                metadata={
                    'command': command,
                    'response': response,
                    'result': result,
                    'success': result.success,
                    'cognitive_updates': cognitive_updates,
                }
            )
        except Exception as e:
            BrainLogger.error(f"Error creating memory data: {e}", exc_info=True)
            return

        BrainLogger.info(f"Memory data for dispatch: {memory_data}")

        try:
            event_payload = memory_data.to_event_data()
            final_event_data = {
                "event_type": "exo_processor",
                "content": event_payload.get("content"),
                "event_data": event_payload
            }
            global_event_dispatcher.dispatch_event(
                Event(f"cognitive:{self.interface_id}:memory_check", final_event_data)
            )
            BrainLogger.info("Event dispatched successfully.")
        except Exception as e:
            BrainLogger.error(f"Error dispatching memory event: {e}", exc_info=True)

            BrainLogger.info(f"Memory data for dispatch: {memory_data}")
            try:
                event_data = memory_data.to_event_data()
                global_event_dispatcher.dispatch_event(
                    Event(f"cognitive:{self.interface_id}:memory_check", event_data)
                )
                BrainLogger.info("Event dispatched successfully.")
            except Exception as e:
                BrainLogger.error(f"Error dispatching memory event: {e}", exc_info=True)
    
    async def format_memory_context(
        self,
        content: Any,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format command interaction context for memory formation.
        
        Args:
            content: LLM response content
            state: Current cognitive state
            metadata: Contains command and result information
        """
        # Ensure metadata is a dictionary.
        metadata = metadata or {}

        command = metadata.get('command')
        result = metadata.get('result')
        
        # Extract command input string.
        if isinstance(command, ParsedCommand):
            command_input = command.raw_input
        elif isinstance(command, dict):
            command_input = command.get('raw_input', 'Unknown command')
        else:
            command_input = 'Unknown command'

        # Extract result message.
        if isinstance(result, CommandResult):
            result_message = result.message
        elif isinstance(result, dict):
            result_message = result.get('message', 'No result')
        else:
            result_message = 'No result'
        
        return get_prompt(
            'interfaces.exo.memory.template',
            model=Config.get_cognitive_model(),
            vars={
                "command_input": command_input,
                "content": content,
                "result_message": result_message,
                'context': metadata.get('cognitive_updates', ''),
            }
        )
        
    async def get_fallback_memory(self, memory_data: MemoryData) -> Optional[str]:
        """
        Generate a fallback memory for the ExoProcessor interface.
        
        Args:
            memory_data: Memory data object containing context and content
        """
        try:
            # Extract core components from memory data
            command = memory_data.metadata.get('command')
            response = memory_data.metadata.get('response', '')
            
            # Format command input
            if isinstance(command, ParsedCommand):
                command_str = command.raw_input
            elif isinstance(command, dict):
                command_str = command.get('raw_input', 'Unknown command')
            else:
                command_str = 'Unknown command'

            # Create a simple structured memory entry
            memory_parts = [
                "Command interaction memory:",
                f"I issued the command: {command_str}",
                f"The response was: {response[:250]}..." if len(response) > 250 else f"The response was: {response}"
            ]
            
            return "\n".join(memory_parts)
            
        except Exception as e:
            BrainLogger.error(f"Error generating fallback memory: {e}")
            return None

    async def get_relevant_memories(self, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to current conversation context.
        """
        if not self.conversation_state or not self.conversation_state.pairs:
            return []
        
        # Get recent context from conversation state
        recent_context = self.conversation_state.get_recent_content(num_pairs=3)
        
        return await self.cognitive_bridge.retrieve_memories(
            query=recent_context,
            limit=3
        )

    async def _get_llm_response(self, context: str) -> str:
        """Get next action from LLM with current context."""
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        # Create temporary history with context
        contextual_messages = self.conversation_state.get_contextual_history(context)

        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=contextual_messages,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )
    
    async def summarize_cognitive_state(self, window_size: int = 20) -> str:
        """
        Create a concise summary of recent cognitive activity.
        Transforms command/response pairs into a coherent narrative.
        """
        summary = None

        if not self.conversation_state or not self.conversation_state.pairs:
            return "no history available; fresh start!"
        
        # Calculate how many pairs to include (2 messages per pair)
        pair_count = min(window_size // 2, len(self.conversation_state.pairs))

        # Format conversation context
        conversation_text = []
        message_list = self.conversation_state.to_message_list()

        # Skip system message (index 1 onwards) then take last N messages
        recent_messages = message_list[1:][-pair_count*2:]

        for msg in recent_messages:
            role = 'Hephia' if msg['role'] == 'assistant' else 'exo'
            content = msg['content'].replace('\n', ' ').strip()
            conversation_text.append(f"{role}: {content}")

        try:
            model_name = Config.get_summary_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            # Create summarization prompt
            sys = get_prompt(
                'interfaces.exo.summary.system',
                model=model_name
            )

            # Get current state for context
            current_state = await self.state_bridge.get_api_context()
            state_summary = TerminalFormatter.format_context_summary(current_state)
            
            user = get_prompt(
                'interfaces.exo.summary.user',
                model=model_name,
                vars={
                    "conversation_history": chr(10).join(conversation_text),
                    "state_summary": state_summary
                }
            )

            summary = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user}
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )

        except Exception as e:
            BrainLogger.error(f"Error generating cognitive summary: {e}")
            
        if not summary:
            try:
                summary = await self.get_fallback_summary(window_size)
            except Exception as e:
                BrainLogger.error(f"Fallback summary also failed: {e}")
    
        return summary if summary else "Failed to generate cognitive summary."
    
    async def get_fallback_summary(self, window_size: int) -> str:
        """Generate a simple fallback summary from recent conversation history."""
        try:
            if not self.conversation_state or not self.conversation_state.pairs:
                return "No conversation history available."
                
            # Get recent messages from conversation state
            message_list = self.conversation_state.to_message_list()
            # Skip system message and get recent messages 
            pair_count = min(window_size // 2, len(self.conversation_state.pairs))
            recent_messages = message_list[1:][-pair_count*2:]

            # Format as IRC style logs
            summary_lines = []
            # Add introductory line matching cognitive continuity theme
            summary_lines.append("Recent cognitive activity trace, your issued commands and responses; maintaining continuity of interaction flow:")
            
            for msg in recent_messages:
                role = 'Hephia' if msg['role'] == 'assistant' else 'ExoProcessor'
                content = msg['content']
                
                # Truncate user messages
                if msg['role'] == 'user':
                    content = content[:250] + ('...' if len(content) > 250 else '')
                    
                summary_lines.append(f"<{role}> {content}")
                
            return "\n".join(summary_lines)
            
        except Exception as e:
            BrainLogger.error(f"Error generating fallback summary: {e}")
            return "Error generating any summary format."

    async def _generate_summary(self, notifications: List[Notification]) -> str:
        """
        Implementation of NotificationManager's abstract _generate_summary.
        Creates a summary of relevant notifications for other interfaces.
        """
        current_time = datetime.now()
        
        # Check if cached summary exists and is still valid
        if (self.cached_summary and self.last_summary_time and 
            (current_time - self.last_summary_time).total_seconds() < self.min_interval):
            return self.cached_summary
            
        try:
            # Generate new summary
            cognitive_summary = await self.summarize_cognitive_state()
            
            # Update cache
            self.cached_summary = f"""Current Internal Cognitive State:
{cognitive_summary}
"""
            self.last_summary_time = current_time
            
            return self.cached_summary
            
        except Exception as e:
            BrainLogger.error(f"Error generating summary: {e}")
            # Return last cached version if available, otherwise error message
            return (self.cached_summary if self.cached_summary 
                   else "Error generating cognitive summary.")

    async def clean_errored_response(self, response: str) -> str:
        """Clean up the LLM response in case of an error."""
        if len(response) <= 150:
            return response.strip()
            
        # Otherwise truncate and clean error message 
        cleaned = response.replace("\n", "\\n").strip()
        truncated = cleaned[:150] + "..."
        return f">>Truncated by ({len(cleaned)}) chars: {truncated}<<"

    async def prune_conversation(self):
        """Remove the last message pair from conversation history and update state."""
        if self.conversation_state.prune_last_exchange():
            BrainLogger.debug("Pruned last conversation pair")
            return True
        else:
            BrainLogger.debug("No pairs to prune")
            return False