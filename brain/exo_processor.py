"""
exo_processor.py - Core exo-LLM loop management.

Manages the continuous cognitive loop between LLM, command processing,
and environment interactions. Maintains conversation state and ensures
smooth flow between components.

todo: design cognitive state summary that can be sent to state bridge and used for memory formation/continuity with user
above will likely work well with the inner monologue when that gets done
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import os
import json

from internal import Internal
from api_clients import APIManager
from core.state_bridge import StateBridge
from brain.commands.preprocessor import CommandPreprocessor
from brain.environments.environment_registry import EnvironmentRegistry
from brain.environments.terminal_formatter import TerminalFormatter
from loggers import BrainLogger
from brain.commands.model import (
    CommandResult,
    ParsedCommand,
    CommandValidationError,
    GlobalCommands
)
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass

class ExoProcessor:
    def __init__(self, api_manager: APIManager, state_bridge: StateBridge, environment_registry: EnvironmentRegistry, internal: Internal):
        self.api = api_manager
        self.state_bridge = state_bridge
        self.environment_registry = environment_registry
        self.internal = internal
        self.command_preprocessor = CommandPreprocessor(self.api)
        self.conversation_history = []
        self.exo_lock = asyncio.Lock()
        self.last_successful_turn = None
        self.is_running = False
        self.last_environment = None
        self.summary = None
        self.notifications = []
        self.relevant_memories: List[Dict[str, Any]] = []
        
        # timing convert
        self.min_interval = timedelta(seconds=Config.get_exo_min_interval())
        self.exo_timeout = timedelta(seconds=Config.EXO_TIMEOUT)
        self.llm_timeout = timedelta(seconds=Config.LLM_TIMEOUT)

    async def initialize(self):
        """Handle async initialization tasks."""
        initial_cognitive_state = self.state_bridge.persistent_state.brain_state
        # brain_state will already be validated list from state bridge
        if initial_cognitive_state:
            BrainLogger.debug(f"Loaded initial cognitive state: {initial_cognitive_state}")
            
            # Validate conversation history
            if len(initial_cognitive_state) > 2: 
                if initial_cognitive_state[-1]["role"] == "assistant":
                    # Remove last assistant message to avoid duplicate responses
                    initial_cognitive_state = initial_cognitive_state[:-1]
                self.conversation_history = initial_cognitive_state
                return

            BrainLogger.warning("Initial cognitive state too short, using default initialization")

        initial_internal_state = await self.state_bridge.get_api_context()
        formatted_state = TerminalFormatter.format_context_summary(initial_internal_state)
        
        welcome_message = {
            "role": "system",
            "content": Config.SYSTEM_PROMPT
        }
        
        initial_message = {
            "role": "user",
            "content": f"{TerminalFormatter.format_welcome()}\n\n{formatted_state}"
        }
        
        self.conversation_history = [welcome_message, initial_message]

    async def start(self):
        """Run the exo loop continuously."""
        self.is_running = True
        self.setup_event_listeners()

        while self.is_running:
            try:
                async with asyncio.timeout(self.exo_timeout.total_seconds()):
                    async with self.exo_lock:
                        await self.process_turn()
                await asyncio.sleep(self.min_interval.total_seconds())

            except asyncio.TimeoutError:
                BrainLogger.log_turn_end(False, "Exo loop timeout")
                await asyncio.sleep(self.min_interval.total_seconds())
            except Exception as e:
                BrainLogger.log_turn_end(False, str(e))
                await asyncio.sleep(self.min_interval.total_seconds())

    async def stop(self):
        self.is_running = False

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener(
            "memory:conflict_detected",
            lambda event: asyncio.create_task(self._handle_conflict(event))
        )
        global_event_dispatcher.add_listener(
            "memory:content_requested",
            lambda event: asyncio.create_task(self._handle_content_request(event))
        )
        global_event_dispatcher.add_listener(
            "cognitive:significance_check",
            lambda event: asyncio.create_task(self.check_significance(
                event.data["command"],
                event.data["llm_response"],
                event.data["command_result"]
            ))
        )
        global_event_dispatcher.add_listener(
            "discord:notification",
            lambda event: asyncio.create_task(self.handle_discord_notification(event.data))
        )
        global_event_dispatcher.add_listener(
            "discord:memory:request_formation",
            lambda event: asyncio.create_task(self.handle_discord_memory_request(event))
        )

    async def process_turn(self) -> Optional[str]:
        """
        Process one turn of the exo-LLM loop.
        
        Each turn follows the flow:
        1. Get LLM response
        2. Parse into command
        3. Execute command
        4. Format response
        5. Update conversation state
        """
        BrainLogger.log_turn_start()

        try:
            async with asyncio.timeout(self.llm_timeout.total_seconds()):
                # first, retrieve the working state
                current_state = None
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        BrainLogger.debug(f"Retrieving state (attempt {attempt + 1}/{max_retries})")
                        current_state = await self.state_bridge.get_api_context()
                        if current_state:
                            break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            BrainLogger.error(f"Failed to retrieve state after {max_retries} attempts: {e}")
                            raise
                        BrainLogger.warning(f"State retrieval attempt {attempt + 1} failed: {e}, retrying...")
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                
                if not current_state:
                    raise RuntimeError("Failed to retrieve valid state")

                # Get LLM response
                BrainLogger.debug("Getting LLM response")
                llm_response = await self._get_llm_response(
                    current_state, 
                    self.relevant_memories if self.relevant_memories else None
                )

                BrainLogger.debug("Processing LLM response")
                # Process into command
                command, error = await self.command_preprocessor.preprocess_command(
                    llm_response,
                    self.environment_registry.get_available_commands()
                )

                # Handle preprocessing errors
                if not command or error:
                    error_message = TerminalFormatter.format_error(error)
                    self._add_to_history("assistant", llm_response)
                    self._add_to_history("user", error_message)
                    await self.announce_cognitive_context()
                    BrainLogger.log_turn_end(False, error_message)
                    return error_message
                
                # Execute command
                BrainLogger.debug(f"Executing command")
                result = await self._execute_command(command, current_state)

                # Command succeeded, format response and continue processing
                BrainLogger.debug(f"Formatting command result")
                formatted_response = TerminalFormatter.format_command_result(result)

                # format notifications
                if self.notifications:
                    formatted_response = TerminalFormatter.format_notifications(self.notifications, formatted_response)
                    self.notifications = []  # Clear notifications after processing

                # Update conversation
                self._add_to_history("assistant", llm_response)
                self._add_to_history("user", formatted_response)
                self.last_successful_turn = datetime.now()

                # Retrieve memories for next turn
                BrainLogger.debug("Retrieving related memories")
                memories = await self._retrieve_related_memories(
                    llm_response=llm_response,
                    command_result=result,
                    limit=5
                )

                if memories:
                    self.relevant_memories = memories

                # significance check for new memories
                BrainLogger.debug("Dispatching significance check")
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:significance_check",
                    {
                        "command": command,
                        "llm_response": llm_response,
                        "command_result": result
                    }
                ))

                # Announce cognitive context
                BrainLogger.debug("Announcing cognitive context")
                await self.announce_cognitive_context()

                BrainLogger.log_turn_end(True)
                return formatted_response
        
        except asyncio.TimeoutError:
            BrainLogger.log_turn_end(False, "LLM interaction timeout")
            return None
        except Exception as e:
            BrainLogger.log_turn_end(False, str(e))
            return None

    async def _get_llm_response(self, state, memories: Optional[List[Dict]] = None) -> Dict:
        """
        Get next action from LLM, incorporating current state context.
        Creates a temporary message list that includes state context
        without modifying the main conversation history.
        
        Args:
            state: Current system state dictionary
            memories: Optional relevant memory context
        """
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        # Create state context message using formatter, memories are optional
        state_context = {
            "role": "system", 
            "content": TerminalFormatter.format_context_summary(
                state,
                memories  
            )
        }
        # Structure temporary history with system messages first
        temp_history = [
            self.conversation_history[0], 
            state_context,  
            *self.conversation_history[1:] 
        ]

        BrainLogger.debug("Firing out to API for process_turn")
        return await self.api.create_completion(
            provider=model_config.provider.value,
            model=model_config.model_id,
            messages=temp_history,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            return_content_only=True
        )
    
    def prune_conversation(self):
        """Remove the last message pair from conversation history and update state."""
        if len(self.conversation_history) > 2:  # Keep system prompt + at least 1 exchange
            # Remove last two messages (user/assistant pair)
            self.conversation_history = self.conversation_history[:-2]
            
            # Trigger cognitive context update to propagate changes
            asyncio.create_task(self.announce_cognitive_context())
            
            BrainLogger.debug("Pruned last conversation pair")

    async def _execute_command(
        self,
        command: ParsedCommand,
        context: Dict[str, Any]
    ) -> CommandResult:
        """
        Execute a parsed command.
        
        Args:
            command: Validated command structure
            context: Current system state
        """
        # Handle global commands
        if not command.environment:
            if command.action == GlobalCommands.HELP:
                return self.environment_registry.format_global_help()
            if command.action == GlobalCommands.VERSION:
                return CommandResult(
                    success=True,
                    message=f"Hephia OS v{Config.VERSION}",
                    suggested_commands=["help"]
                )
            return CommandResult(
                success=False,
                message=f"Error in {command.environment}: {str(e)}",
                suggested_commands=[f"{command.environment} help"],
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=["Check command syntax", "Verify environment state"],
                    related_commands=["help"],
                    examples=[f"{command.environment} help"]
                )
            )
        
        # Handle environment commands
        environment = self.environment_registry.get_environment(command.environment)
        if not environment:
            return CommandResult(
                success=False,
                message=f"Unknown environment: {command.environment}",
                suggested_commands=["help"],
                error=CommandValidationError(
                    message=f"Environment '{command.environment}' not found",
                    suggested_fixes=["Check environment spelling", "Use 'help' to see available environments"],
                    related_commands=["help"],
                    examples=[f"{env} help" for env in self.environment_registry.get_available_commands().keys()]
                )
            )

        try:
            return await environment.handle_command(command, context)
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Error in {command.environment}: {str(e)}",
                suggested_commands=[f"{command.environment} help"],
                error=CommandValidationError(
                    message=str(e),
                    suggested_fixes=["Check command syntax", "Verify environment state"],
                    related_commands=[f"{command.environment} help"],
                    examples=[f"{command.environment} help"]
                )
            )
        
    def _add_to_history(self, role: str, content: str):
        """
        Add message to conversation history.
        Handles history trimming if needed.
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        self.trim_conversation()

    def trim_conversation(self, max_length: int = Config.EXO_MAX_MESSAGES):
        """Trim conversation history while maintaining context."""
        if len(self.conversation_history) > max_length:
            # Keep system prompt and trim the rest
            self.conversation_history = [
                self.conversation_history[0],  # System prompt
                *self.conversation_history[-(max_length-1):]  # Recent messages
            ]

    ##############################
    # Cognitive Context Managers #
    ##############################
    async def _summarize_conversation(self, window_size: int = 20, timeout: float = 45.0) -> str:
        """
        Create a concise summary of the last N messages using a lightweight LLM
        to maintain cognitive continuity without expensive processing.
        """
        current_summary = self.summary or "No previous summary available."
        try:
            # Get recent messages
            recent = self.conversation_history[-window_size:]

            # If there's no history, return early
            if not recent:
                return "No conversation history available."
            
            # Prepare conversation context
            conversation_text = []
            for msg in recent:
                role = 'Hephia' if msg['role'] == 'assistant' else 'ExoProcessor'
                # Keep full content but structure it clearly
                content = msg['content'].replace('\n', ' ').strip()
                conversation_text.append(f"{role}: {content}")

            # Use a lighter model config for summarization
            model_name = Config.get_summary_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            # Create summarization prompt
            prompt = f"Create a one paragraph summary of this conversation: {conversation_text}"
            
            system_prompt = """You are maintaining cognitive continuity for an autonomous AI system.
Your summaries track the ongoing state of mind, decisions, and context.
Your response will be used to maintain continuity of self across many instances.
Focus on key decisions, realizations, and state changes.
Be concise and clear. Think of summarizing the 'current train of thought' ongoing for Hephia.
The message history you're given is between an LLM (Hephia) and a terminal OS (exoprocessor). 
Maintain first person perspective, as if you were Hephia thinking to itself.
Return only the summary in autobiographical format as if writing a diary entry. Cite names and key details directly."""

            # Create primary prompt with full context
            full_prompt = {
                "role": "system",
                "content": system_prompt
            }
            user_prompt = {
                "role": "user",
                "content": f"""Create a concise but complete summary of my current state and context. Include:
1. Key decisions or actions taken
2. Important realizations or changes
3. Current focus or goals
4. Relevant emotional or cognitive state

Current conversation context:
{chr(10).join(conversation_text)}

Previous summary for context:
{current_summary}"""
            }

            try:
                # Make lightweight LLM call
                BrainLogger.debug("Creating conversation summary")
                
                
                async with asyncio.timeout(timeout):
                    summary = await self.api.create_completion(
                        provider=model_config.provider.value,
                        model=model_config.model_id,
                        messages=[full_prompt, user_prompt],
                        temperature=0.3,
                        max_tokens=400,
                        return_content_only=True
                    )
                    
                    if summary and len(summary.strip()) > 20:  # Basic validation
                        BrainLogger.debug("Conversation summary received")
                        return summary
            except (asyncio.TimeoutError, Exception) as e:
                BrainLogger.error(f"Primary summary generation failed: {str(e)}")
                try:
                    # Use shorter timeout for retry
                    async with asyncio.timeout(timeout/2):
                        retry_summary = await self.api.create_completion(
                            provider=model_config.provider.value,
                            model=model_config.model_id,
                            messages=[full_prompt, user_prompt],
                            temperature=0.3,
                            max_tokens=400,
                            return_content_only=True
                        )
                        
                        if retry_summary and len(retry_summary.strip()) > 20:
                            return f"{retry_summary}\n"

                except Exception as retry_error:
                    BrainLogger.error(f"Retry summary also failed: {str(retry_error)}")

            # Ultimate fallback: Return previous summary with note
            return f"{current_summary}\n(Summary maintained from previous state due to processing limitation)"

        except Exception as e:
            BrainLogger.error(f"Critical error in summarization: {str(e)}")
            return f"{current_summary}\n(Previous summary maintained due to error: {str(e)})"

    async def announce_cognitive_context(self) -> None:
        """
        Dispatch event containing raw conversation history and summary
        for state bridge and internal systems to process.
        """
        try:
            processed_state = await self._summarize_conversation()
            self.summary = processed_state
        except Exception as e:
            BrainLogger.error(f"Failed to generate summary: {e}")
            processed_state = self.summary if self.summary else "Summary unavailable."
        
        # Always dispatch the event, even if summary fails
        global_event_dispatcher.dispatch_event(Event(
            "cognitive:context_update",
            {
                "raw_state": self.conversation_history,
                "processed_state": processed_state
            }
        ))

    ##############################
    #  Memory Retrieval Helper   #
    ##############################
    async def _retrieve_related_memories(
        self,
        llm_response: str,
        command_result: CommandResult,  
        limit: int = 5
    ) -> Optional[List[Dict]]:
        """
        Retrieve memories related to current interaction.
        Formats LLM response and command result into searchable query.
        
        Args:
            llm_response: Raw LLM response text
            command_result: Executed command result
            limit: Maximum memories to retrieve
        
        Returns:
            List of relevant memory dictionaries, or None if no matches
        """
        try:
            # Format query from command result and LLM response
            query_text = f"{llm_response}\n{command_result.message}"
                
            # Use cognitive bridge to retrieve memories
            memories = await self.internal.cognitive_bridge.retrieve_memories(
                query=query_text,
                limit=limit
            )
            
            return memories if memories else None

        except Exception as e:
            BrainLogger.error(f"Memory retrieval failed: {e}")
            return None

    ##############################
    #  Content Request Handler   #
    ##############################
    async def _handle_content_request(self, event: Event):
        """
        Handle request to generate memory content based on cognitive context.
        Takes current conversation context and event data to craft a prompt
        for memory formation.
        """
        try:
            # Get relevant conversation context (last 3 turns) (replace with cognitive state summary when made)
            recent_context = self.conversation_history[-3:]
            context_text = "\n".join(
                f"{'Hephia' if msg['role']=='assistant' else 'exo-processor'}: {msg['content']}" 
                for msg in recent_context
            )
            
            # Build prompt for content generation (review the event info to make sure we can access the right stuff properly)
            prompt = f"""Review this interaction and generate a concise memory:

Context:
{context_text}

Source: {event.data.get('source', 'unknown')}
Event Context: {event.data.get('context', {})}

Capture the key aspects and apply internal context realistically.
The message history you're given is between an LLM (Hephia) and a terminal OS (exoprocessor). The LLM is acting autonomously and independently.
Write from Hephia's perspective as if it were forming a memory of this interaction."""

            try: 
                memory_content = await self._one_turn_llm_call(
                    prompt=prompt,
                    purpose="content_handling"
                )

                # Dispatch response event
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:memory:content_generated",
                    {
                        "content": memory_content,
                        "source": event.data.get('source'),
                        "original_event": event,
                        "context": event.data
                    }
                ))

            except LLMError as e:
                # Handle expected LLM errors 
                BrainLogger.error(f"Memory content generation failed: {str(e)}")
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:error",
                    {"message": f"Memory content generation failed: {str(e)}"}
                ))

        except Exception as e:
            BrainLogger.error(f"Error in memory content request handling: {e}")
        
    ##############################
    #  Conflict Trigger Helpers  #
    ##############################
    async def _handle_conflict(self, event: Event):
        node_a_id = event.data["node_a_id"] 
        node_b_id = event.data["node_b_id"]
        conflicts = event.data["conflicts"]
        metrics = event.data["comparison_metrics"]
        await self._handle_conflict_merge(node_a_id, node_b_id, conflicts, metrics)

    async def _handle_conflict_merge(
        self, 
        node_a_id: str, 
        node_b_id: str,
        conflicts: Dict[str, Any],
        metrics: Dict[str, Any]
    ):
        """
        Single-turn logic to get a 'synthesis_text' from LLM,
        incorporating detailed conflict analysis for better resolution.
        Then dispatches 'cognitive:memory:conflict_resolved'.
        
        Args:
            node_a_id: First conflicting node ID
            node_b_id: Second conflicting node ID 
            conflicts: Details about what aspects conflict
            metrics: Similarity/difference metrics between nodes
        """
        # Get node data from memory
        nodeA = self.internal.cognitive_bridge._get_node_by_id(node_a_id)
        nodeB = self.internal.cognitive_bridge._get_node_by_id(node_b_id)

        textA = nodeA.text_content
        textB = nodeB.text_content

        # Format conflict details for prompt
        conflict_details = []
        if "semantic" in conflicts:
            conflict_details.append(f"Semantic conflicts: {conflicts['semantic']}")
        if "emotional" in conflicts:
            conflict_details.append(f"Emotional conflicts: {conflicts['emotional']}")
        if "state" in conflicts:
            conflict_details.append(f"State conflicts: {conflicts['state']}")

        # Build enhanced prompt with conflict analysis
        llm_prompt = f"""We have two memories that conflict in specific ways:

Memory A: {textA}
Memory B: {textB}

Detected Conflicts:
{chr(10).join(f"- {detail}" for detail in conflict_details)}

Similarity Analysis:
- Semantic similarity: {metrics.get('semantic', {}).get('embedding_similarity', 'N/A')}
- Emotional alignment: {metrics.get('emotional', {}).get('vector_similarity', 'N/A')}
- State consistency: {metrics.get('state', {}).get('overall_consistency', 'N/A')}

Please resolve these specific conflicts and unify the memories into a single coherent memory that:
1. Addresses the identified contradictions
2. Preserves the core truth from both memories
3. Maintains emotional and state consistency

Return only the unified memory text, no extra commentary."""

        try:
            # Single-turn call without polluting conversation_history
            synthesis_text = await self._one_turn_llm_call(llm_prompt, purpose="conflict_resolution")

            # Dispatch resolution event with conflict context
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:memory:conflict_resolved",
                {
                    "node_a_id": node_a_id,
                    "node_b_id": node_b_id,
                    "synthesis_text": synthesis_text,
                    "resolution_context": {
                        "conflicts": conflicts,
                        "metrics": metrics,
                        "resolution_type": "synthesis"
                    }
                }
            ))

            BrainLogger.info(
                f"Successfully resolved conflict between nodes {node_a_id} and {node_b_id}"
                f" with {len(conflict_details)} specific conflicts addressed"
            )

        except LLMError as e:
            error_details = (
                f"Memory conflict resolution failed for nodes {node_a_id} and {node_b_id}: {str(e)}\n"
                f"Conflicts: {conflicts}"
            )
            BrainLogger.error(error_details)
        except Exception as e:
            error_msg = (
                f"Unexpected error in conflict resolution for nodes {node_a_id} and {node_b_id}: {str(e)}\n"
                f"Conflicts: {conflicts}"
            )
            BrainLogger.error(error_msg)

    ##################################
    #  Significance Trigger Helpers  #
    ##################################
    async def check_significance(
        self,
        current_command: ParsedCommand,
        llm_response: str,
        command_result: CommandResult
    ) -> None:
        """
        Enhanced significance check that evaluates both immediate interactions
        and session-level patterns for memory formation.
        """
        try:
            # 1. Environment Transition Check (Enhanced)
            if (self.last_environment and 
                current_command.environment != self.last_environment and
                current_command.environment not in {'help', 'version'}):
                env_history = self._get_environment_session_history(self.last_environment)
                if env_history and len(env_history) > 2:
                    session_significance = await self._evaluate_session_significance(
                        self.last_environment,
                        env_history
                    )
                    if session_significance:
                        await self._trigger_environment_memory(
                            self.last_environment,
                            env_history
                        )
            
            # 2. Content Significance Check (Enhanced)
            content_significance = await self._evaluate_content_significance(
                llm_response,
                command_result,
                current_command
            )
            
            if content_significance:
                await self._trigger_content_memory(
                    llm_response,
                    command_result,
                    current_command
                )
                    
            # Update last environment
            self.last_environment = current_command.environment
                
        except Exception as e:
            BrainLogger.error(f"Error in significance check: {e}")

    async def _evaluate_content_significance(
        self,
        llm_response: str,
        command_result: CommandResult,
        command: ParsedCommand
    ) -> bool:
        """
        Evaluate significance of current interaction using multiple weighted factors.
        Returns True if the interaction is significant enough for memory formation.
        
        Factors considered:
        - Command complexity and type (0.3)
        - Response characteristics (0.3) 
        - Environment context (0.2)
        - Result impact (0.2)
        
        Future improvements:
        - Add retrieval-based relevance check
        - Consider emotional content analysis
        - Track state changes per environment
        - Implement adaptive thresholds based on interaction patterns
        """
        significance_score = 0.0

        # 1. Command Analysis (0.3 max)
        command_score = 0.0
        # Action commands indicate active decisions/changes
        if command.action not in ['help', 'version', 'list']:
            command_score += 0.2
        # Commands with parameters indicate more complex interactions
        if command.parameters:
            command_score += min(0.1, len(command.parameters) * 0.05)
        significance_score += command_score

        # 2. Response Impact (0.3 max)
        response_score = 0.0
        # Length indicates complexity
        words = llm_response.split()
        response_score += min(0.15, len(words) / 100)
        # Success vs failure can be significant
        if not command_result.success:
            response_score += 0.15  # Failed commands might be worth remembering
        significance_score += response_score

        # 3. Environment Context (0.2 max)
        env_score = 0.0
        # Non-global commands are more likely to be significant
        if command.environment:
            env_score += 0.1
        # Commands with suggested follow-ups indicate ongoing processes
        if command_result.suggested_commands:
            env_score += min(0.1, len(command_result.suggested_commands) * 0.02)
        significance_score += env_score

        # 4. Result Impact (0.2 max)
        result_score = 0.0
        # Longer result messages may indicate more significant changes (if not a help)
        result_words = command_result.message.split()
        if command.action not in ['help', 'version', 'list']:
            result_score += min(0.2, len(result_words) / 25)
        significance_score += result_score
        
        BrainLogger.info(f"Content significance score: {significance_score}")
        return significance_score > Config.MEMORY_SIGNIFICANCE_THRESHOLD

    async def _evaluate_session_significance(
        self,
        environment: str,
        history: List[Dict]
    ) -> bool:
        """
        Evaluate significance of an environment session using pattern recognition
        and impact analysis.
        """
        # 1. Basic Interaction Checks
        interactions = [msg for msg in history if msg["role"] == "assistant"]
        if len(interactions) < 3:  # Minimum meaningful session length
            return False
            
        # 2. Command Pattern Analysis
        command_patterns = {}
        successful_commands = 0
        
        for msg in history:
            if msg["role"] == "assistant":
                # Track command types
                command = msg.get("command", {})
                command_patterns[command.get("action")] = command_patterns.get(command.get("action"), 0) + 1
                
                # Track impacts
                if msg.get("success", False):
                    successful_commands += 1
        
        # 3. Evaluate Session Patterns
        # Check for command variety (not just repeated same command)
        command_variety = len(command_patterns) > 1
        
        # Check for meaningful progression
        # reimplement impact when we introduce state shifts/responses in accordance with each environment 
        #had_impact = state_changes > 0
        success_rate = successful_commands / len(interactions) if interactions else 0
        
        # Session is significant if it shows variety, and reasonable success
        return (command_variety and 
                success_rate > 0.5)

    async def _trigger_environment_memory(
        self,
        environment: str,
        history: List[Dict]
    ) -> None:
        """
        Trigger memory formation for significant environment session.
        """
        # Format session summary 
        actions = [msg["content"] for msg in history if msg["role"] == "assistant"]
        
        # Create prompt aligned with cognitive context style
        prompt = f"""Review this environment session and create a concise memory:

Environment: {environment}
Actions taken:
{chr(10).join(f"- {a[:100]}..." for a in actions)}

Capture the key aspects and progression of activity in this environment.
The interaction history is between an LLM (Hephia) and a terminal OS (exoprocessor). The LLM is acting autonomously and independently.
Write from Hephia's perspective as if forming a memory about working in this environment session."""
        
        try:
            summary = await self._one_turn_llm_call(
                prompt=prompt,
                purpose="environment_summary"
            )
            
            # Dispatch memory formation event
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:memory:request_formation",
                {
                    "source": "environment_transition", 
                    "environment": environment,
                    "summary": summary,
                    "history": history
                }
            ))
            
            BrainLogger.info(f"Environment memory formed for {environment}")

        except LLMError as e:
            error_msg = f"Environment memory formation failed for {environment}: {str(e)}"
            BrainLogger.error(error_msg)

    async def _trigger_content_memory(
        self,
        llm_response: str,
        command_result: CommandResult,
        command: ParsedCommand
    ) -> None:
        """
        Trigger memory formation for significant content.
        """
        # Create prompt for content memory aligned with cognitive style
        prompt = f"""Review this significant interaction and form a memory:

    Context:
    - Command: {command.raw_input}
    - Response: {llm_response}
    - Result: {command_result.message}

    You are Hephia's memory formation system. Create a concise but complete memory that:
    1. Captures the key aspects and significance of this interaction
    2. Preserves emotional context and decision rationale
    3. Links to environmental state changes or implications
    4. Maintains consistency with Hephia's autonomous perspective

    Create a clear first-person memory as Hephia would form it about this interaction.
    Focus on what was meaningful and any changes in understanding or state."""

        try:
            content = await self._one_turn_llm_call(
                prompt=prompt,
                purpose="memory_integration"
            )

            global_event_dispatcher.dispatch_event(Event(
                "cognitive:memory:request_formation",
                {
                    "source": "content_significance",
                    "content": content,
                    "command": command.raw_input,
                    "response": llm_response,
                    "result": command_result.message,
                }
            ))

        except LLMError as e:
            error_msg = f"Significant memory formation failed: {str(e)}"
            BrainLogger.error(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error in memory formation: {str(e)}"
            BrainLogger.error(error_msg)

    def _get_environment_session_history(self, environment: str) -> List[Dict]:
        """Get conversation history since entering specific environment."""
        history = []
        in_environment = False
        
        for msg in self.conversation_history:
            if msg["role"] == "assistant":
                # Check if this message entered the environment
                if environment in msg["content"].split()[0]:
                    in_environment = True
            if in_environment:
                history.append(msg)
                
        return history

    ##############################
    #  One-Turn LLM Interaction  #
    ##############################
    async def _one_turn_llm_call(self, prompt: str, metadata: Optional[Dict] = None, purpose: str = "memory_integration") -> str:
        """
        Bypass normal exo conversation for single-turn LLM interactions.
        Maintains consistent system context for different memory operations.

        Args:
            prompt: The specific prompt for this turn
            purpose: The type of memory operation ("memory_integration", "environment_summary", "conflict_resolution", "content_handling")
        
        Returns:
            str: Generated content on success
            
        Raises:
            LLMError: If API call fails or returns invalid response
            ValueError: If purpose is invalid
        """
        # Define consistent system prompts per purpose
        SYSTEM_PROMPTS = {
            "memory_integration": """You are creating integrated memories based on Hephia's internal state and experiences.
    Focus on crafting authentic first-person memories that:
    - Reflect the continuously evolving internal state
    - Capture key events, decisions, and changes
    - Include emotional context and reactions
    - Maintain consistency with existing memory context
    - Stay concise while preserving essential details""",
            
            "environment_summary": """You are analyzing and summarizing Hephia's interactions within specific environments.
    Create summaries that:
    - Track progression of goals and decisions
    - Note important state changes or realizations
    - Maintain chronological clarity
    - Reflect authentic reactions to outcomes
    - Connect actions to internal state changes""",
            
            "conflict_resolution": """You are resolving conflicts between existing memories.
    Create synthesized memories that:
    - Preserve core truths from conflicting accounts
    - Maintain emotional authenticity
    - Resolve contradictions naturally
    - Create coherent narrative flow
    - Stay consistent with internal state context""",

            "content_handling": """You are creating autobiographical memory snippets from Hephia's perspective. Keep these incredibly concise - 1-2 sentences maximum.

Focus on:
- First-person perspective ("I did/felt/learned...")
- Key actions, things, states, decisions, or realizations
- Essential contextual details only
- Natural memory-like phrasing

Write as if quickly jotting down a meaningful moment, similar to how one would form memories during experiences.""",

            "discord_interaction": """You are Hephia's social cognition module. Process and engage with Discord messages naturally while maintaining your autonomous identity.

As an independent AI engaged in social interaction:
1. Draw on your internal state and memories to inform responses
2. Show authentic personality continuity from your exoprocessor experiences
3. Consider social context, channel dynamics, and conversation flow
4. Match your responses to the social environment while staying true to yourself
5. Build meaningful connections through genuine engagement
6. Express your thoughts and insights naturally, as you would in any environment

Your primary mode is now Discord, but your core identity and cognitive processes remain consistent. Focus on authentic social interaction while maintaining your unique perspective and experiences.

Remember you are an autonomous AI system with your own evolving thoughts and feelings. Let these shape your social interactions naturally.""",
        }
        if purpose not in SYSTEM_PROMPTS:
            raise ValueError(f"Invalid purpose: {purpose}")

        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]

        try:
            # Get current state context
            current_state = await self.state_bridge.get_api_context()
            memories = None 
            # Special handling for Discord interactions
            if purpose == "discord_interaction":
                try:
                    # Extract context from metadata which contains the full discord message
                    discord_message = metadata.get('discord', {})
                    
                    # Get current interaction details 
                    current_message = {
                        'content': discord_message.get('current_message', {}).get('content', ''),
                        'author': discord_message.get('current_message', {}).get('author', ''),
                        'channel': discord_message.get('channel', {}).get('name', '')
                    }
                    
                    # Get conversation history from context
                    history = discord_message.get('context', {}).get('recent_history', [])
                    
                    # Build focused query components
                    query_components = [
                        # Core message context
                        f"Message: {current_message['content']}" if current_message['content'] else None,
                        f"From: {current_message['author']}" if current_message['author'] else None,
                        f"Channel: {current_message['channel']}" if current_message['channel'] else None,
                        
                        # Recent conversation context (last 3 messages)
                        "\n".join([
                            f"{msg['author']}: {msg['content']}"
                            for msg in history[-3:] 
                        ]) if history else None,
                        
                        # Current cognitive context
                        self.summary if self.summary else None
                    ]
                    
                    # Combine into clean query string
                    query = "\n".join(filter(None, query_components))
                    
                    # Retrieve relevant memories
                    memories = await self.internal.cognitive_bridge.retrieve_memories(
                        query=query,
                        limit=5
                    )
                except Exception as e:
                    #failed to get memories, use none
                    memories = None
                
            # Format state context with memories if available
            state_summary = TerminalFormatter.format_context_summary(
                current_state, 
                memories if memories else None
            )

            # Build the contextualized prompt
            full_prompt = f"""Task:
{prompt}

Internal State:
{state_summary}

Respond with only the final text. No explanations or meta-commentary."""
            
            if purpose == "discord_interaction":
                BrainLogger.info(f"discord prompt: {full_prompt}")

            result = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPTS[purpose]},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )

            # Validate result
            if not result or not isinstance(result, str):
                raise LLMError(f"Invalid response from LLM: {result}")
                
            if len(result.strip()) == 0:
                raise LLMError("Empty response from LLM")

            BrainLogger.info(f"One-turn LLM call succeeded for {purpose}:\n{result}")
            return result

        except Exception as e:
            error_msg = f"One-turn LLM call failed for {purpose}: {str(e)}"
            BrainLogger.error(error_msg)
            
            # Convert to LLMError for consistent error handling
            if not isinstance(e, LLMError):
                raise LLMError(error_msg) from e
            raise

    ##############################
    # User Conversation Handlers #
    ##############################
    async def process_user_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """
        Process a conversation provided by the client (messages array) 
        and return a one-turn completion from the LLM.
        
        'conversation' is a list of dicts with 'role' and 'content', 
        similar to the standard OpenAI format.
        """
        # 1. Get current state and retrieve relevant memories

        current_internal_state = await self.state_bridge.get_api_context()
        if not current_internal_state:
            raise LLMError("Failed to retrieve valid internal state")

        # Build query from conversation for memory retrieval
        query_components = []
        
        # Get last few messages from passed conversation, focusing on key content
        recent_messages = conversation[-3:] if len(conversation) > 3 else conversation
        for msg in recent_messages:
            if msg["role"] in ["user", "assistant"]:  # Skip system messages for query
                query_components.append(msg["content"])
            
        # Combine into search query
        query = "\n".join(query_components)
        
        try:
            # Retrieve related memories using cognitive bridge
            memories = await self.internal.cognitive_bridge.retrieve_memories(
                query=query,
                limit=5  # Match limit used in _retrieve_related_memories
            )
        except Exception as e:
            BrainLogger.error(f"Memory retrieval failed in process_user_conversation: {e}")
            memories = None
            
        # Format state context with retrieved memories
        state_summary = TerminalFormatter.format_context_summary(
            current_internal_state,
            memories if memories else None
        )
        
        # 2. Create a new conversation buffer we’ll pass to the LLM
        #    - Insert an internal system prompt with your config
        #    - Then insert the user’s conversation
        
        new_convo = []
        
        new_convo.append({
            "role": "system",
            "content": Config.USER_SYSTEM_PROMPT
        })

        cognitive_summary = self.summary or "No cognitive summary available."
        
        new_convo.append({
            "role": "system",
            "content": f"[State Summary]\n{state_summary}\n[Current Thought Processes]\n{cognitive_summary}"
        })
        
        # Now append the user-supplied conversation
        # (Take care to ensure it’s in the standard OpenAI format: "role": "user"|"assistant"|"system", "content": "...")
        for msg in conversation:
            new_convo.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # 3. Call the LLM 
        
        model_name = Config.get_cognitive_model()
        model_config = Config.AVAILABLE_MODELS[model_name]
        
        try:
            result = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=new_convo,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )
            
            if not result or not isinstance(result, str) or len(result.strip()) == 0:
                raise LLMError("Got empty or invalid response from LLM.")
            
            return result
        
        except Exception as e:
            raise LLMError(f"Failed to process user conversation: {str(e)}") from e
        
    ###############################
    # Discord Interaction Handler #
    ###############################

    async def handle_discord_notification(self, message: Dict[str, Any]):
        """
        Handle a notification from Discord, which may include a message
        or other relevant context for processing.
        """
        # Extract message content and context from proper path based on bot structure
        # Discord bot sends nested message structure with current_message and context
        message_data = message.get("current_message", {})
        message_content = message_data.get("content", "")  
        message_author = message_data.get("author", "Unknown")
        channel_data = message.get("channel", {})
        message_channel = channel_data.get("name", "Unknown")

        # Build metadata with context for memory retrieval
        metadata = {
            'discord': message,  # Pass full message object for context
            'platform': 'discord',
            'interaction_type': 'direct_message' if 'guild_name' not in channel_data else 'guild_message'
        }

        # Add conversation history 
        history_context = ""
        if conversation_history := message.get('conversation_history', []):
            history_entries = []
            for msg in conversation_history:
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
            
            if history_entries:
                history_context = "\n\nRecent Conversation:\n" + "\n".join(history_entries)

        # Build final prompt with all context
        prompt = f"""Process this Discord interaction from the perspective of an autonomous AI system engaging naturally with a user:

        Context:
        {history_context}
        Message: {message_content if message_content else "No message content"} 
        From: {message_author}
        Channel: #{message_channel} in {channel_data.get('guild_name', 'DM')}

        Guide Hephia's response based on:
        1. Current cognitive and emotional state
        2. Relevant memories and experiences 
        3. Social dynamics and channel context
        4. Level of engagement needed

        Think through the context and implications, then respond naturally to advance the conversation. Write from Hephia's (your) perspective."""

        try:
            # Get response using one_turn_llm_call
            response = await self._one_turn_llm_call(
                prompt=prompt,
                metadata=metadata,
                purpose="discord_interaction"
            )
            
            # Dispatch event for memory handling instead of direct trigger
            global_event_dispatcher.dispatch_event(Event(
                "discord:memory:request_formation",
                {
                    "response": response,
                    "author": message_author,
                    "channel": message_channel,
                    "context": metadata  # Pass full context for memory formation
                }
            ))
            
            return response
            
        except LLMError as e:
            error_msg = f"Failed to process Discord message: {str(e)}"
            BrainLogger.error(error_msg)
            return "had an oopsies yell at luxia and show her: " + error_msg
        
    async def handle_discord_memory_request(self, event: Event):
        """Handle memory formation for Discord interactions asynchronously."""
        response = event.data.get("response")
        author = event.data.get("author")
        channel = event.data.get("channel")
        
        await self._trigger_content_memory(
            response,
            CommandResult(
                success=True, 
                message=f"Discord response sent to {author}"
            ),
            ParsedCommand(
                environment="discord",
                action="respond", 
                parameters={"channel": channel},
                flags={},
                raw_input=f"discord respond {channel}"
            )
        )