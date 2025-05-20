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

    async def retrieve_memories(self, query: str, limit: int = 5, context_state: Optional[Dict[str, Any]] = None, threshold: float = 0) -> List[Dict[str, Any]]:
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
                return_details=True
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

    async def absorb_memory(self, memory_id: str, intensity: float) -> Dict[str, Any]:
        """
        Deep focus on specific memory with amplified echo effects.
        Abstracts memory interaction through the memory system orchestrator.
        """
        try:
            # Use memory system's meditation method instead of direct manipulation
            meditation_result = await self.memory_system.meditate_on_memory(
                memory_id=memory_id,
                intensity=intensity,
                duration=5  # Default duration for absorption
            )

            if not meditation_result:
                return None

            # Format result using meditation data
            result = {
                "content": meditation_result.get("content"),
                "effects": ["Memory becoming clearer"],
                "connected_effects": []
            }

            # Process connected memories from meditation result
            if "connected_memories" in meditation_result:
                for connection in meditation_result["connected_memories"]:
                    result["connected_effects"].append({
                        "content": connection["content"],
                        "strength": connection["connection_strength"],
                        "depth": connection["depth"]
                    })

            # Include echo effects if available
            if "echo_effects" in meditation_result:
                result["effects"].extend(meditation_result["echo_effects"])

            return result

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Memory absorption failed: {str(e)}"}
            ))
            return None

    async def _handle_memory_formation(self, event: Event):
        """Handle memory formation requests in a source-agnostic way."""
        try:
            event_data = event.data.get('event_data', {})
            content = event_data.get('content')
            context = event_data.get('context')
            
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