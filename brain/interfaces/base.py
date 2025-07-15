"""
brain/interfaces/base.py

Base classes for cognitive interfaces, providing common functionality
for context formatting, notification management, and cognitive continuity.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any
from api_clients import APIManager
from loggers import BrainLogger
from config import Config
from event_dispatcher import global_event_dispatcher, Event
from brain.utils.tracer import brain_trace
from brain.cognition.notification import Notification, NotificationManager, NotificationInterface
from brain.cognition.memory.significance import MemoryData
from brain.environments.terminal_formatter import TerminalFormatter
from brain.prompting.loader import get_prompt

from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge

class CognitiveInterface(NotificationInterface, ABC):
    def __init__(
        self, 
        interface_id: str,
        state_bridge: StateBridge,
        cognitive_bridge: CognitiveBridge,
        notification_manager: NotificationManager,
        api_manager: APIManager
    ):
        super().__init__(interface_id, notification_manager)
        self.state_bridge = state_bridge
        self.cognitive_bridge = cognitive_bridge
        self.api = api_manager

    @abstractmethod
    async def process_interaction(self, content: Any) -> Any:
        """Process an interaction through this interface."""
        pass

    @abstractmethod
    async def format_memory_context(
        self,
        content: Any,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format context specifically for memory formation.
        Uses same context as regular interactions but formatted for memory creation.
        """
        pass

    @abstractmethod
    async def get_fallback_memory(self, memory_data: MemoryData) -> Optional[str]:
        """Get a fallback memory for this interface."""
        pass
    
    @brain_trace
    async def get_cognitive_context(self, metadata: Optional[Dict[str, Any]] = None) -> AsyncGenerator[tuple[str, Any], None]:
        """Get formatted cognitive context including state and notifications, yielding each piece."""
        brain_trace.context.memory("Getting relevant memories")
        memories = await self.get_relevant_memories(metadata=metadata) or []
        yield ("memories", memories)
        BrainLogger.debug(f"[{self.interface_id}] Got memories: {memories}")

        brain_trace.context.state("Retrieving API context")
        state = await self.state_bridge.get_api_context() or {}
        yield ("state", state)
        BrainLogger.debug(f"[{self.interface_id}] Got API context: {state}")

        brain_trace.context.format("Formatting cognitive context")
        formatted_context = TerminalFormatter.format_context_summary(state, memories) or "No context available"
        yield ("formatted_context", formatted_context)
        
        brain_trace.context.notifications("Getting updates from other interfaces")
        other_updates = await self.notification_manager.get_updates_for_interface(self.interface_id)
        BrainLogger.info(f"[{self.interface_id}] Got updates from other interfaces: {other_updates}")
        yield ("updates", other_updates)
    
    @abstractmethod
    async def get_relevant_memories(self, metadata: Optional[Dict[str, any]] = None) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to this interface's current context."""
        pass

    async def announce_cognitive_context(self, raw, notification: Notification) -> None:
        """Notify event system about new cognitive context."""
        try:
            # Get our own summary to announce
            summary = await self.notification_manager._summary_formatters[self.interface_id]([notification])  # Pass relevant notifications
            
            await global_event_dispatcher.dispatch_event_async(Event(
                "cognitive:context_update",
                {
                    "source": self.interface_id,
                    "raw_state": raw,
                    "processed_state": summary
                }
            ))
            BrainLogger.info(f"[{self.interface_id}] Context announcement complete")
        except Exception as e:
            BrainLogger.error(f"Error announcing cognitive context for {self.interface_id}: {e}")

    async def analyze_cognitive_influence(
        self, 
        formatted_response: str, 
        other_updates: str,
        metadata: Optional[Dict[str, Any]] = None,
        notification: Optional[Notification] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze cognitive influence from interface interaction using interface-agnostic summary.
        
        Args:
            formatted_response: The formatted response from this interaction
            other_updates: Updates from other interfaces (cognitive context)
            metadata: Additional context (command, result, etc.)
            notification: Optional notification to use for influence analysis
            
        Returns:
            List of influence dictionaries with valence, arousal, intensity, name, source_type
        """
        try:
            brain_trace.interaction.cognitive_influence("Analyzing cognitive influence")
            
            if notification is None:
                # guarantee notification for summary
                notification = Notification(
                    content=self._create_influence_content(formatted_response, metadata),
                    source_interface=self.interface_id,
                    timestamp=datetime.now()
                )
            
            # Get interface-agnostic summary using existing pattern
            summary = await self._generate_summary([notification])
            
            # Combine with other updates for full cognitive context
            full_content = f"{summary}\n\nSystem Context:\n{other_updates}"
            
            # Try LLM analysis first
            influences = await self._llm_cognitive_influence_analysis(full_content)
            if influences:
                BrainLogger.info(f"[{self.interface_id}] LLM cognitive influence analysis successful: {len(influences)} influences")
                return influences
                
            # Fallback to heuristic analysis
            BrainLogger.warning(f"[{self.interface_id}] LLM analysis failed, using heuristic fallback")
            return self._fallback_cognitive_influence_analysis(formatted_response, metadata)
            
        except Exception as e:
            BrainLogger.error(f"Error in cognitive influence analysis for {self.interface_id}: {e}")
            # Emergency fallback - return neutral influence
            return [{
                'name': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'intensity': 0.1,
                'source_type': 'cognitive_fallback',
                'source_data': {'error': str(e)}
            }]
        
    def _create_influence_content(
        self, 
        formatted_response: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create content for influence analysis notification.
        Interface-specific implementation can be overridden if needed.
        """
        return {
            'response': formatted_response,
            'metadata': metadata or {},
            'interface_type': self.interface_id,
            'analysis_type': 'cognitive_influence'
        }
    
    async def _llm_cognitive_influence_analysis(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Use LLM to analyze cognitive influence from content summary.
        
        Args:
            content: Interface-agnostic summary + system context
            
        Returns:
            List of influence dictionaries or None if analysis fails
        """
        try:
            # Use summary model for quick analysis
            model_name = Config.get_summary_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            # Build analysis prompt
            system_prompt = get_prompt(
                "interfaces.influence.system",
                model=model_name
            )
            user_prompt = get_prompt(
                "interfaces.influence.user",
                model=model_name,
                vars={
                    "content": content
                }
            )
            
            # Get LLM response
            response = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=200,
                return_content_only=True
            )
            
            if response:
                return self._parse_influence_response(response)
            return None
            
        except Exception as e:
            BrainLogger.error(f"LLM cognitive influence analysis failed: {e}")
            return None
        
    def _parse_influence_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response with INFLUENCE: format into influence dictionaries.
        
        Expected format: INFLUENCE: [name]|[valence]|[arousal]|[intensity]
        Example: INFLUENCE: accomplished|0.4|0.2|0.6
        
        Args:
            response: Raw response content from LLM
            
        Returns:
            List of influence dictionaries
        """
        influences = []
        try:
            lines = response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Look for INFLUENCE: lines (case insensitive)
                if line.upper().startswith('INFLUENCE:'):
                    try:
                        # Remove the INFLUENCE: prefix
                        influence_data = line[10:].strip()  # Remove "INFLUENCE:" (10 chars)
                        
                        # Split by pipe and clean whitespace
                        parts = [part.strip() for part in influence_data.split('|')]
                        
                        if len(parts) != 4:
                            BrainLogger.warning(f"Invalid influence format (expected 4 parts): {line}")
                            continue
                        
                        name, valence_str, arousal_str, intensity_str = parts
                        
                        # Parse and validate numerical values
                        try:
                            valence = max(-1.0, min(1.0, float(valence_str)))
                            arousal = max(-1.0, min(1.0, float(arousal_str)))  
                            intensity = max(0.0, min(1.0, float(intensity_str)))
                        except ValueError as num_error:
                            BrainLogger.warning(f"Invalid numerical values in influence: {line} - {num_error}")
                            continue
                        
                        # Skip very weak influences
                        if intensity < 0.05:
                            BrainLogger.debug(f"Skipping weak influence: {name} (intensity: {intensity})")
                            continue
                        
                        # Clean name and validate
                        name = name.strip().lower().replace(' ', '_')
                        if not name:
                            name = 'unnamed'
                        
                        influences.append({
                            'name': name,
                            'valence': valence,
                            'arousal': arousal,
                            'intensity': intensity,
                            'source_type': 'cognitive_influence',
                            'source_data': {
                                'interface': self.interface_id,
                                'analysis_method': 'llm',
                                'raw_line': line.strip(),
                                'parsed_from': 'pipe_format'
                            }
                        })
                        
                    except (ValueError, IndexError) as parse_error:
                        BrainLogger.warning(f"Failed to parse influence line '{line}': {parse_error}")
                        continue
                
                # Handle common variations/typos
                elif any(variant in line.upper() for variant in ['INFLUENCE-', 'INFLUENCE ', 'INFLUENCES:']):
                    BrainLogger.debug(f"Found potential influence line with variant format: {line}")
                    # Try to salvage by normalizing
                    normalized = line.upper().replace('INFLUENCE-', 'INFLUENCE:').replace('INFLUENCE ', 'INFLUENCE:').replace('INFLUENCES:', 'INFLUENCE:')
                    # Recursively parse the normalized line
                    nested_result = self._parse_influence_response(normalized)
                    influences.extend(nested_result)
            
            if influences:
                BrainLogger.info(f"Successfully parsed {len(influences)} cognitive influences from LLM response")
            else:
                BrainLogger.debug("No valid influences found in LLM response")
                BrainLogger.debug(f"Raw response: {response}")
                
            return influences
            
        except Exception as e:
            BrainLogger.error(f"Error parsing cognitive influence response: {e}")
            BrainLogger.debug(f"Response that failed to parse: {response}")
            return []
        
    def _fallback_cognitive_influence_analysis(
        self,
        formatted_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple rule-based cognitive influence analysis fallback.
        
        Args:
            formatted_response: The response content to analyze
            metadata: Additional context for analysis
            
        Returns:
            List of influence dictionaries
        """
        influences = []
        content_lower = formatted_response.lower()
        
        # Success/accomplishment patterns
        if any(word in content_lower for word in ['completed', 'successful', 'found', 'created', 'accomplished']):
            influences.append({
                'name': 'accomplished',
                'valence': 0.3,
                'arousal': 0.1,
                'intensity': 0.4,
                'source_type': 'cognitive_heuristic',
                'source_data': {'pattern': 'success', 'interface': self.interface_id}
            })
        
        # Learning/discovery patterns  
        if any(word in content_lower for word in ['learned', 'discovered', 'insight', 'understanding', 'realized']):
            influences.append({
                'name': 'curious',
                'valence': 0.2,
                'arousal': 0.4,
                'intensity': 0.5,
                'source_type': 'cognitive_heuristic',
                'source_data': {'pattern': 'discovery', 'interface': self.interface_id}
            })
        
        # Challenge/problem patterns
        if any(word in content_lower for word in ['error', 'failed', 'problem', 'issue', 'difficulty']):
            influences.append({
                'name': 'challenged',
                'valence': -0.2,
                'arousal': 0.3,
                'intensity': 0.4,
                'source_type': 'cognitive_heuristic',
                'source_data': {'pattern': 'challenge', 'interface': self.interface_id}
            })
        
        # If no patterns matched, return minimal neutral influence
        if not influences:
            influences.append({
                'name': 'neutral',
                'valence': 0.0,
                'arousal': 0.0,
                'intensity': 0.1,
                'source_type': 'cognitive_heuristic',
                'source_data': {'pattern': 'neutral', 'interface': self.interface_id}
            })
        
        return influences