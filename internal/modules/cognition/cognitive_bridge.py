"""
cognitive_bridge.py

Manages interaction between conscious and subconscious systems.
Handles reflection requests, meditation effects, and memory traversal while maintaining
clean separation between mechanical and cognitive layers.
"""

import asyncio
from typing import Dict, List, Any, Optional
from event_dispatcher import Event, global_event_dispatcher
from internal.modules.memory.memory_system import MemorySystemOrchestrator
from internal.modules.memory.nodes.cognitive_node import CognitiveMemoryNode
from ...internal_context import InternalContext

import time

from loggers.loggers import BrainLogger

class CognitiveBridge:
    """
    Bridges conscious and subconscious processes, orchestrating memory traversal,
    reflection, and meditation effects without direct state manipulation.
    """
    
    def __init__(self, internal_context: InternalContext, MemorySystemOrchestrator: MemorySystemOrchestrator):
        """
        Initialize the bridge with required system references.

        Args:
            internal_context: Access to system state
            MemorySystemOrchestrator: Access to memory system
        """
        self.internal_context = internal_context
        self.memory_system = MemorySystemOrchestrator
        self.cognitive_state = {"raw": [], "processed": {}}
        self.logger = BrainLogger
        self.setup_event_listeners()

    def setup_event_listeners(self):
        global_event_dispatcher.add_listener(
            "cognitive:memory:request_formation",
            lambda event: asyncio.create_task(self._handle_memory_formation(event))
        )
        global_event_dispatcher.add_listener(
            "cognitive:context_update",
            self.update_cognitive_state
        )
        global_event_dispatcher.add_listener(
            "significance:request_metrics_evaluation",
            lambda event: asyncio.create_task(self._handle_significance_evaluation_request(event))
        )


    def update_cognitive_state(self, event: Event):
        try:
            self.cognitive_state['raw'] = event.data.get('raw_state', [])
            self.cognitive_state['processed'] = event.data.get('processed_state', {})
        except Exception as e:
            print(f"Error updating cognitive state: {e}")
            raise

    def get_cognitive_state(self) -> Dict[str, Any]:
        if not self.cognitive_state.get("raw_state") or self.cognitive_state.raw is None:
            return {"raw_state": [], "processed_state": "System initializing"} 
        else:
            return {"raw_state": self.cognitive_state.raw, "processed_state": self.cognitive_state.processed}

    async def _get_node_by_id(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """
        Helper method to get a cognitive memory node by ID.
        
        Args:
            node_id: ID of node to retrieve

        Returns:
            Optional[CognitiveMemoryNode]: The requested node if found, None otherwise
        """
        return await self.memory_system.get_node_by_id(node_id)

    async def reflect_on_topic(self, topic: str, depth: int) -> List[Dict[str, Any]]:
        """
        Search memory for nodes related to topic, triggering echo effects.
        """
        try:
            # Get memory context state
            context_state = await self.internal_context.get_memory_context(is_cognitive=True)
            
            # Use the memory system's reflection method
            reflection_results = await self.memory_system.reflect_on_memories(
                topic=topic,
                depth=depth,
                context_state=context_state
            )

            # Guard against None or empty results
            if not reflection_results:
                return []

            # Format results for API consistency
            results = []
            for reflection in reflection_results:
                try:
                    memory_data = {
                        'id': reflection.get('node_id'),
                        'content': reflection.get('content'),
                        'timestamp': reflection.get('timestamp'),
                        'relevance': reflection.get('relevance', 0.0),
                        'connections': []
                    }
                    
                    # Safely process connections
                    if 'connected_memories' in reflection:
                        connected = reflection['connected_memories']
                        if isinstance(connected, list):
                            memory_data['connections'] = [
                                {
                                    'id': conn.get('node_id'),
                                    'content': conn.get('content', 'No content available'),
                                    'weight': conn.get('connection_strength', 0.0),
                                    'depth': conn.get('depth', 1)
                                }
                                for conn in connected
                                if isinstance(conn, dict)
                            ]
                    
                    # Only append if we have valid content
                    if memory_data.get('content'):
                        results.append(memory_data)
                        
                except Exception as node_error:
                    self.logger.error(f"Error processing reflection node: {node_error}")
                    continue

            return results

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Reflection failed: {str(e)}"}
            ))
            self.logger.error(f"Reflection error: {str(e)}")
            return []

    async def retrieve_memories(self,
                                query: str,
                                limit: int = 5,
                                context_state: Optional[Dict[str, Any]] = None,
                                threshold: float = 0,
                                dispatch_echo: bool = True) -> List[Dict[str, Any]]:
        """
        Helper method to retrieve memories using cognitive memory's retrieval system.

        Args:
            query: Search query/topic
            limit: Maximum number of memories to return
            context_state: State context for matching (optional)

        Returns:
            List[Dict]: List of formatted memory results
        """
        try:
            if not context_state:
                context_state = await self.internal_context.get_memory_context(is_cognitive=True)

            # Use memory system's cognitive retrieval method
            memories = await self.memory_system.retrieve_cognitive_memories(
                query=query,
                comparison_state=context_state,
                top_k=limit,
                threshold=threshold,
                return_details=True,
                dispatch_echo=dispatch_echo
            )
            
            if not memories or not memories[0]:  # Check both memories and metrics
                return []

            nodes, metrics = memories
            
            # Format results with metrics data
            results = []
            for node, metric_data in zip(nodes, metrics):
                memory_data = {
                    'id': node.node_id,
                    'content': node.text_content,
                    'timestamp': node.timestamp,
                    'strength': node.strength,
                    'relevance': metric_data['final_score']
                }
                results.append(memory_data)

            return results

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error", 
                {"message": f"Memory retrieval failed: {str(e)}"}
            ))
            return []

    def _format_relative_time(self, time_diff: float) -> str:
        """
        Format timestamp into human-readable relative time, rounded to nearest 5.
        Helper method for memory formatting.
        """
        minutes = time_diff / 60
        hours = minutes / 60
        days = hours / 24
        
        if days >= 1:
            return f"about {round(days / 5) * 5} days ago"
        if hours >= 1:
            return f"about {round(hours / 5) * 5} hours ago"
        if minutes >= 1:
            return f"about {round(minutes / 5) * 5} minutes ago"
        return "moments ago"

    async def get_recent_memories(self, limit: int) -> List[Dict[str, Any]]:
        """
        Retrieve and echo recent cognitive memories.
        """
        try:
            # Use async method from memory system
            recent = await self.memory_system.get_recent_memories(
                count=limit,
                network_type="cognitive",
                include_ghosted=False
            )
            
            if not recent:
                return []

            results = []
            context_state = await self.internal_context.get_memory_context(is_cognitive=True)

            for node in recent:
                if not isinstance(node, CognitiveMemoryNode):
                    continue

                # Request echo effect through event system
                global_event_dispatcher.dispatch_event(Event(
                    "memory:echo_requested",
                    {
                        "node_id": node.node_id,
                        "similarity": 0.75,  # Base recall intensity
                        "given_state": context_state,
                        "query_text": "recent_memory_recall",
                        "query_embedding": None,
                        "precalculated_metrics": None
                    }
                ))

                # Format with rounded timestamp
                time_diff = time.time() - node.timestamp
                formatted_time = self._format_relative_time(time_diff)
                
                results.append({
                    'id': node.node_id,
                    'content': node.text_content,
                    'time': formatted_time,
                    'strength': node.strength,
                    'source': node.formation_source
                })

            return results

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Recent memory retrieval failed: {str(e)}"}
            ))
            return []

    def meditate_on_state(self, state: str, intensity: float, duration: int) -> Dict[str, Any]:
        """
        Generate multi-system meditation effects.
        """
        try:
            BrainLogger.info(f"Meditating on state: {state} ({intensity}, {duration})")
            intensity = max(0.0, min(1.0, intensity))
            duration = max(1, min(10, duration))
            effects = []

            # Emotional effects (can simply generate new emotions in the processor. all should, honestly.)
            if state in ['calm', 'relaxed', 'peaceful']:
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:emotional:meditation",
                    {
                        "type": "calming",
                        "intensity": intensity,
                        "duration": duration
                    }
                ))
                effects.append("Deepening sense of peace")

            elif state in ['focused', 'alert', 'aware']:
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:emotional:meditation",
                    {
                        "type": "focusing",
                        "intensity": intensity,
                        "duration": duration
                    }
                ))
                effects.append("Sharpening awareness")

            # Mood effects (modifies overall mood vectors)
            if state in ['happy', 'joyful', 'content']:
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:mood:meditation",
                    {
                        "type": "elevation",
                        "intensity": intensity,
                        "duration": duration
                    }
                ))
                effects.append("Mood brightening")

            elif state in ['melancholy', 'somber', 'reflective']:
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:mood:meditation",
                    {
                        "type": "deepening",
                        "intensity": intensity,
                        "duration": duration
                    }
                ))
                effects.append("Mood becoming contemplative")

            # If no effects generated yet, dispatch subtle calming effect
            if not effects:
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:emotional:meditation",
                    {
                        "type": "calming",
                        "intensity": intensity * 0.3,  # Reduced intensity
                        "duration": duration
                    }
                ))
                effects.append("Subtle calming")

            # need effects (to be done later; need to make a way to have temporary effects, like with echo. eventually cause decaying shifts to the base/multiplicative rate)

            # Return meditation results
            return {
                "intensity": "strong" if intensity > 0.7 else "moderate" if intensity > 0.3 else "subtle",
                "direction": "deepening",
                "effects": effects
            }

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Meditation failed: {str(e)}"}
            ))
            return {}
        
    async def focus_on_state_with_memories(
        self, 
        state_query: str, 
        intensity: float = 0.5, 
        duration: int = 1
    ) -> Dict[str, Any]:
        """
        Focus meditation that finds memories related to the state and uses their
        emotional patterns to create more targeted meditation effects.
        
        Args:
            state_query: Abstract state or feeling to focus on
            intensity: Desired meditation intensity 
            duration: Meditation duration in cycles
            
        Returns:
            Dict containing meditation results and emotional patterns found
        """
        try:
            # Get current context
            context = await self.internal_context.get_memory_context(is_cognitive=True)
            
            # Do mini-reflect to find relevant memories (smaller set than full reflect)
            related_memories = await self.retrieve_memories(
                query=state_query,
                limit=5,  # Smaller set for focus vs absorb
                context_state=context
            )
            
            if not related_memories:
                # Fall back to basic meditation if no memories found
                return self.meditate_on_state(state_query, intensity, duration)
            
            # Extract emotional patterns from memories
            emotional_patterns = []
            total_relevance = 0.0
            
            for memory in related_memories:
                relevance = memory.get('relevance', 0.0)
                total_relevance += relevance
                
                # Try to extract emotional context from memory if available
                # This would come from the memory's processed_state when it was formed
                if 'emotional_state' in memory.get('processed_state', {}):
                    emotions = memory['processed_state']['emotional_state']
                    for emotion in emotions:
                        emotional_patterns.append({
                            'valence': emotion.get('valence', 0.0),
                            'arousal': emotion.get('arousal', 0.0),
                            'intensity': emotion.get('intensity', 0.0),
                            'weight': relevance
                        })
            
            if not emotional_patterns:
                # If no emotional patterns found, fall back to basic meditation
                return self.meditate_on_state(state_query, intensity, duration)
            
            # Calculate weighted average emotional direction
            total_weight = sum(p['weight'] for p in emotional_patterns)
            if total_weight > 0:
                avg_valence = sum(p['valence'] * p['weight'] for p in emotional_patterns) / total_weight
                avg_arousal = sum(p['arousal'] * p['weight'] for p in emotional_patterns) / total_weight
                avg_intensity = sum(p['intensity'] * p['weight'] for p in emotional_patterns) / total_weight
            else:
                avg_valence = avg_arousal = avg_intensity = 0.0
            
            # Create memory-informed meditation events
            memory_informed_effects = []
            
            # Dispatch emotional meditation based on discovered patterns
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:emotional:meditation",
                {
                    "type": "memory_informed",
                    "state_query": state_query,
                    "valence_direction": avg_valence,
                    "arousal_direction": avg_arousal,
                    "intensity": intensity * (1 + avg_intensity * 0.5),  # Boost based on memory intensity
                    "duration": duration,
                    "memory_count": len(related_memories)
                }
            ))
            
            memory_informed_effects.append(f"Drawing from {len(related_memories)} related memories")
            memory_informed_effects.append(f"Emotional resonance: {avg_valence:.2f} valence, {avg_arousal:.2f} arousal")
            
            # Also dispatch mood meditation if the pattern suggests it
            if abs(avg_valence) > 0.2:
                mood_type = "elevation" if avg_valence > 0 else "deepening"
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:mood:meditation",
                    {
                        "type": mood_type,
                        "intensity": intensity * abs(avg_valence),
                        "duration": duration
                    }
                ))
                memory_informed_effects.append(f"Mood influence: {mood_type}")
            
            return {
                "state_query": state_query,
                "memory_count": len(related_memories),
                "emotional_patterns": emotional_patterns,
                "average_direction": {
                    "valence": avg_valence,
                    "arousal": avg_arousal,
                    "intensity": avg_intensity
                },
                "effects": memory_informed_effects,
                "total_relevance": total_relevance
            }
            
        except Exception as e:
            self.logger.error(f"Memory-informed focus meditation error: {str(e)}")
            # Fall back to basic meditation on error
            return self.meditate_on_state(state_query, intensity, duration)

    async def absorb_memory(
        self, 
        query: str, 
        intensity: float = 0.8, 
        max_memories: int = 3
    ) -> Dict[str, Any]:
        """
        Absorb memories based on a query topic rather than specific memory ID.
        Retrieves relevant memories and performs meditation on the top results.
        
        Args:
            query: Topic or theme to search for memories about
            intensity: Meditation intensity for each memory
            max_memories: Maximum number of memories to absorb
            
        Returns:
            Dict containing absorbed memories and their collective effects
        """
        try:
            # Get current context for memory retrieval
            context = await self.internal_context.get_memory_context(is_cognitive=True)
            
            # Retrieve relevant memories using existing infrastructure
            relevant_memories = await self.retrieve_memories(
                query=query,
                limit=max_memories,
                context_state=context,
                dispatch_echo=False  # No need to dispatch echoes here, done in meditation
            )
            
            if not relevant_memories:
                return None
                
            # Meditate on each retrieved memory
            absorbed_memories = []
            collective_effects = []
            
            for memory in relevant_memories:
                meditation_result = await self.memory_system.meditate_on_memory(
                    memory_id=memory['id'],
                    intensity=intensity
                )
                
                if meditation_result and 'error' not in meditation_result:
                    absorbed_memories.append({
                        'memory_id': memory['id'],
                        'content': memory['content'],
                        'relevance': memory.get('relevance', 0.0),
                        'effects': meditation_result.get('echo_effects', [])
                    })
                    
                    # Collect effects for summary
                    if meditation_result.get('echo_effects'):
                        collective_effects.extend(meditation_result['echo_effects'])
            
            if not absorbed_memories:
                return None
                
            return {
                'query': query,
                'absorbed_count': len(absorbed_memories),
                'memories': absorbed_memories,
                'collective_effects': collective_effects,
                'intensity': intensity
            }
            
        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Memory absorption by query failed: {str(e)}"}
            ))
            self.logger.error(f"Query-based absorption error: {str(e)}")
            return None

    async def _handle_memory_formation(self, event: Event):
        """Handle memory formation requests in a source-agnostic way."""
        try:
            event_data = event.data.get('event_data', {})
            
            # Request memory formation through the memory system
            global_event_dispatcher.dispatch_event(Event(
                "memory:formation_requested",
                {
                    "event_type": event.data.get('event_type', 'unknown'),
                    "event_data": event_data
                }
            ))
        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Memory formation failed: {str(e)}"}
            ))

    async def _handle_significance_evaluation_request(self, event: Event) -> None:
        """
        Handle significance evaluation requests from SignificanceAnalyzer.
        Coordinates with memory system to perform sophisticated metrics evaluation.
        
        Expected event data:
        - request_id: Unique identifier for this evaluation request
        - memory_data: MemoryData object as dict
        - generated_content: LLM-generated memory content
        - timeout: Maximum evaluation time
        """
        request_id = None
        try:
            request_id = event.data.get("request_id")
            memory_data_dict = event.data.get("memory_data", {})
            generated_content = event.data.get("generated_content", "")
            timeout = event.data.get("timeout", 10.0)
            
            if not request_id:
                self.logger.error("Significance evaluation request missing request_id")
                return
                
            if not generated_content:
                self.logger.warning(f"Significance evaluation {request_id} has no generated content")
                await self._send_significance_response(
                    request_id=request_id,
                    significance_score=0.0,
                    component_scores={},
                    evaluation_method="error",
                    error="No generated content provided"
                )
                return
            
            self.logger.debug(f"Processing significance evaluation request {request_id}")
            
            # Perform the metrics-based evaluation
            evaluation_result = await self._evaluate_memory_significance(
                memory_data_dict=memory_data_dict,
                generated_content=generated_content,
                timeout=timeout
            )
            
            # Send response back to SignificanceAnalyzer
            await self._send_significance_response(
                request_id=request_id,
                **evaluation_result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle significance evaluation request {request_id}: {e}")
            if request_id:
                await self._send_significance_response(
                    request_id=request_id,
                    significance_score=0.0,
                    component_scores={},
                    evaluation_method="error",
                    error=str(e)
                )

    async def _evaluate_memory_significance(
        self,
        memory_data_dict: Dict[str, Any],
        generated_content: str,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Coordinate with memory system to evaluate memory significance using sophisticated metrics.
        Creates temporary node and uses existing metrics infrastructure.
        
        Args:
            memory_data_dict: MemoryData as dictionary
            generated_content: LLM-generated memory content
            timeout: Maximum evaluation time
            
        Returns:
            Dict containing significance_score, component_scores, evaluation_method, error
        """
        try:
            # Get current memory context for comparison
            current_context = await self.internal_context.get_memory_context(is_cognitive=True)
            if not current_context:
                return {
                    "significance_score": 0.5,
                    "component_scores": {},
                    "evaluation_method": "fallback",
                    "error": "No memory context available"
                }
            
            # Use memory system's evaluation infrastructure
            # This leverages the same temporary node + metrics pattern from _calculate_initial_strength
            significance_score = await self._delegate_to_memory_system_evaluation(
                generated_content=generated_content,
                current_context=current_context,
                source_type=memory_data_dict.get("source_type", "unknown"),
                timeout=timeout
            )
            
            # For now, return a simplified response
            # The detailed component scores could be added later if needed
            return {
                "significance_score": significance_score,
                "component_scores": {
                    "overall": significance_score
                },
                "evaluation_method": "metrics",
                "error": None
            }
            
        except Exception as e:
            self.logger.error(f"Memory significance evaluation failed: {e}")
            return {
                "significance_score": 0.5,
                "component_scores": {},
                "evaluation_method": "error_fallback", 
                "error": str(e)
            }
        
    async def _delegate_to_memory_system_evaluation(
        self,
        generated_content: str,
        current_context: Dict[str, Any],
        source_type: str,
        timeout: float
    ) -> float:
        """
        Delegate to memory system for actual significance evaluation.
        Now uses the dedicated evaluate_memory_significance method.
        
        Args:
            generated_content: LLM-generated memory content
            current_context: Current memory/cognitive context  
            source_type: Type of memory source (for logging)
            timeout: Maximum evaluation time
            
        Returns:
            float: Significance score between 0.0 and 1.0
        """
        try:
            # Use the new dedicated significance evaluation method
            significance_score = await self.memory_system.evaluate_memory_significance(
                generated_content=generated_content,
                context=current_context,
                source_type=source_type,
                timeout=timeout
            )
            
            self.logger.debug(f"Memory system significance evaluation: {significance_score:.3f} for {source_type}")
            return significance_score
            
        except Exception as e:
            self.logger.error(f"Memory system significance evaluation error for {source_type}: {e}")
            return 0.5  # Neutral score on error

    async def _send_significance_response(
        self,
        request_id: str,
        significance_score: float,
        component_scores: Dict[str, float],
        evaluation_method: str,
        error: Optional[str] = None
    ) -> None:
        """
        Send significance evaluation response back to SignificanceAnalyzer.
        
        Args:
            request_id: Unique identifier matching the original request
            significance_score: Overall significance score (0.0 - 1.0)
            component_scores: Breakdown by component (semantic, emotional, etc.)
            evaluation_method: How the evaluation was performed
            error: Error message if evaluation failed
        """
        try:
            response_data = {
                "request_id": request_id,
                "significance_score": significance_score,
                "component_scores": component_scores,
                "evaluation_method": evaluation_method
            }
            
            if error:
                response_data["error"] = error
                
            global_event_dispatcher.dispatch_event(Event(
                "significance:metrics_evaluation_response",
                response_data
            ))
            
            self.logger.debug(f"Sent significance response for {request_id}: {significance_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to send significance response for {request_id}: {e}")