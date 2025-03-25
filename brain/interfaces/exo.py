"""
brain/interfaces/exo_interface.py

Implements the ExoProcessor interface for the command-based terminal environment.
Handles command processing, LLM interactions, and maintenance of cognitive continuity
for the core command loop while integrating with the broader cognitive architecture.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from brain.utils.tracer import brain_trace
from brain.interfaces.base import CognitiveInterface
from brain.cognition.notification import Notification, NotificationManager
from brain.core.command_handler import CommandHandler
from core.state_bridge import StateBridge
from brain.commands.model import CommandResult, ParsedCommand, GlobalCommands
from brain.cognition.memory.significance import MemoryData, SourceType
from brain.environments.terminal_formatter import TerminalFormatter
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
    ):
        super().__init__("exo_processor", state_bridge, cognitive_bridge, notification_manager)
        self.api = api_manager
        self.command_handler = command_handler
        self.conversation_history = []
        self.last_successful_turn = None
        self.last_summary_time = None
        self.cached_summary = ""
        
        # Timing configuration
        self.min_interval = Config.get_exo_min_interval()
        self.llm_timeout = Config.LLM_TIMEOUT

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
                # Get cognitive context
                brain_trace.interaction.context("Getting cognitive context")
                context_data = {}
                async for key, value in self.get_cognitive_context():
                    context_data[key] = value
                
                context = context_data.get("formatted_context", "No context available")
                other_updates = context_data.get("updates", [])
                
                BrainLogger.info(f"[{self.interface_id}] Received cognitive context: {context}")
                brain_trace.interaction.context.complete(context=context)
                
                # Get LLM response
                brain_trace.interaction.llm("Getting LLM response")
                llm_response = await self._get_llm_response(context)
                
                # Process command
                brain_trace.interaction.command("Processing command")
                command, error = await self.command_handler.preprocess_command(llm_response)
                
                if error:
                    # cleaning only for when hallucinated context is terribly long
                    cleaned_response = await self.clean_errored_response(llm_response)
                    brain_trace.interaction.error(error_msg=error)
                    self._add_to_history("assistant", cleaned_response)
                    error_message = TerminalFormatter.format_error(error)
                    self._add_to_history("user", error_message)
                    return error_message
                
                # Execute command
                brain_trace.interaction.execute("Executing command")
                result = await self.command_handler.execute_command(command)
                
                # Format response
                brain_trace.interaction.format("Formatting response")
                formatted_response = TerminalFormatter.format_command_result(result)
                BrainLogger.debug(f"Terminal Response: {formatted_response}")
                final_response = TerminalFormatter.format_notifications(other_updates, formatted_response)
                BrainLogger.debug(f"Attached updates: {other_updates}")
                BrainLogger.debug(f"Final Response: {final_response}")
                # Update conversation
                brain_trace.interaction.update("Updating conversation state")

                #just return the llm response; the final_response should carry all information relating to any processing *we* did.
                self._add_to_history("assistant", llm_response)
                self._add_to_history("user", final_response)
                self.last_successful_turn = datetime.now()

                # Log conversation history to file
                log_path = "data/logs/exo_conversation.log"
                try:
                    with open(log_path, "w", encoding="utf-8") as f:
                        for msg in self.conversation_history:
                            f.write(f"{msg['role']}: {msg['content']}\n")
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

                await self.announce_cognitive_context(self.conversation_history, notification)
                await self._dispatch_memory_check(formatted_response, command, result)
                
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
                    "conversation_length": len(self.conversation_history),
                    "last_successful": self.last_successful_turn.isoformat() if self.last_successful_turn else None
                }
            )
            return None
        
    async def _dispatch_memory_check(self, response: str, command: ParsedCommand, result: CommandResult) -> None:
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
                    'success': result.success
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
        
    async def initialize(self, brain_state = None) -> None:
        """Initialize the exo processor interface and conversation history."""
        if len(brain_state) > 2: 
                if brain_state[-1]["role"] == "assistant":
                    # Remove last assistant message to avoid duplicate responses
                    brain_state = brain_state[:-1]
                self.conversation_history = brain_state
                return
            
        initial_state = await self.state_bridge.get_api_context()
        formatted_state = TerminalFormatter.format_context_summary(initial_state)
        
        # Initial system prompt and welcome message
        welcome_message = {
            "role": "system",
            "content": Config.SYSTEM_PROMPT
        }
        
        initial_message = {
            "role": "user", 
            "content": f"{TerminalFormatter.format_welcome()}\n\n{formatted_state}"
        }
        
        if not self.conversation_history:
            self.conversation_history = [welcome_message, initial_message]
        
        # Reset timing trackers
        self.last_successful_turn = None
    
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
        
        return f"""Review this significant interaction and form a memory:

Context:
- Command: {command_input}
- Response: {content}
- Result: {result_message}

Create a clear first-person memory as Hephia would form it about this interaction.
Focus on what was meaningful and any changes in understanding or state."""

    async def format_cognitive_context(
        self,
        state: Dict[str, Any],
        memories: List[Dict[str, Any]]
    ) -> str:
        """
        Format cognitive context for command processing.
        Includes state context and relevant memories.
        """
        return TerminalFormatter.format_context_summary(state, memories)

    async def get_relevant_memories(self, metadata: Optional[str]) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to current conversation context.
        """
        if not self.conversation_history:
            return []
            
        # Get recent context
        recent_context = "\n".join(
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-3:]
        )
        
        return await self.cognitive_bridge.retrieve_memories(
            query=recent_context,
            limit=3
        )

    async def _get_llm_response(self, context: str) -> str:
        """Get next action from LLM with current context."""
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        # Create temporary history with context
        temp_history = [
            self.conversation_history[0],  # System prompt
            {"role": "system", "content": context},
            *self.conversation_history[1:]
        ]
        
        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=temp_history,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )
    
    async def summarize_cognitive_state(self, window_size: int = 20) -> str:
        """
        Create a concise summary of recent cognitive activity.
        Transforms command/response pairs into a coherent narrative.
        """
        try:
            if not self.conversation_history:
                return "No conversation history available."
                
            # Get recent messages
            recent = self.conversation_history[-min(window_size, len(self.conversation_history)):]
            if not recent:
                return "No recent cognitive activity."
            
            # Format conversation context
            conversation_text = []
            for msg in recent:
                role = 'Hephia' if msg['role'] == 'assistant' else 'ExoProcessor'
                content = msg['content'].replace('\n', ' ').strip()
                conversation_text.append(f"{role}: {content}")
            
            model_name = Config.get_summary_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            # Create summarization prompt
            system_prompt = """You are maintaining cognitive continuity for an autonomous AI system.
Your summaries track the ongoing state of mind, decisions, and context.
Your response will be used to maintain continuity of self across many instances.
Focus on key decisions, realizations, and state changes.
Be concise and clear. Think of summarizing the 'current train of thought' ongoing for Hephia.
The message history you're given is between an LLM (Hephia) and a terminal OS (exoprocessor). 
Maintain first person perspective, as if you were Hephia thinking to itself.
Return only the summary in autobiographical format as if writing a diary entry. Cite names and key details directly."""

            # Get current state for context
            current_state = await self.state_bridge.get_api_context()
            state_summary = TerminalFormatter.format_context_summary(current_state)
            
            user_prompt = f"""Create a concise but complete summary of my current state and context. Include:
1. Key decisions or actions taken
2. Important realizations or changes
3. Current focus or goals
4. Relevant emotional or cognitive state

Current conversation context:
{chr(10).join(conversation_text)}

Current state context:
{state_summary}"""

            summary = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=400,
                return_content_only=True
            )
            
            return summary if summary else "Failed to generate cognitive summary."
            
        except Exception as e:
            BrainLogger.error(f"Error generating cognitive summary: {e}")
            return "Error in cognitive summary generation."

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
            self.cached_summary = f"""Current Cognitive State:
{cognitive_summary}
"""
            self.last_summary_time = current_time
            
            return self.cached_summary
            
        except Exception as e:
            BrainLogger.error(f"Error generating summary: {e}")
            # Return last cached version if available, otherwise error message
            return (self.cached_summary if self.cached_summary 
                   else "Error generating cognitive summary.")

    def _add_to_history(self, role: str, content: str) -> None:
        """Add message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        self._trim_conversation()

    def _trim_conversation(self, max_length: int = Config.EXO_MAX_MESSAGES) -> None:
        """Trim conversation history while maintaining a rolling window of most recent messages.
        Ensures the last message is always a user message for proper LLM prompting."""
        
        if len(self.conversation_history) > max_length:
            # Always keep system message (index 0) and max_length-1 most recent messages
            self.conversation_history = [
                self.conversation_history[0],  # Keep system prompt
                *self.conversation_history[-(max_length-1):]  # Keep most recent messages
            ]
            
            # Validate conversation pairs are intact (user followed by assistant)
            # Start from beginning after system message
            i = 1
            while i < len(self.conversation_history) - 1:
                current = self.conversation_history[i]
                next_msg = self.conversation_history[i + 1]
                
                if (current["role"] == "user" and next_msg["role"] == "assistant"):
                    i += 2  # Move to next pair
                else:
                    # Found unpaired message, remove it
                    self.conversation_history.pop(i)
                    
            # Final check - if last message is assistant, remove it to ensure user message is last
            if len(self.conversation_history) > 1:
                if self.conversation_history[-1]["role"] == "assistant":
                    self.conversation_history.pop()
                    
            # If we somehow end up with an assistant message at the end after all that,
            # keep removing messages until we get to a user message
            while (len(self.conversation_history) > 1 and 
                   self.conversation_history[-1]["role"] == "assistant"):
                self.conversation_history.pop()

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
        if len(self.conversation_history) > 2:  # Keep system prompt + at least 1 exchange
            # Remove last two messages (user/assistant pair)
            self.conversation_history = self.conversation_history[:-2]
            
            BrainLogger.debug("Pruned last conversation pair")