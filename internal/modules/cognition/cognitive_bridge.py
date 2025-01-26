"""
cognitive_bridge.py

Manages interaction between conscious (ExoProcessor/LLM) and subconscious systems.
Handles reflection requests, meditation effects, and memory traversal while maintaining
clean separation between mechanical and cognitive layers.
"""

import asyncio
from typing import Dict, List, Any, Optional
from event_dispatcher import Event, global_event_dispatcher
from internal.modules.memory.cognitive_memory_node import CognitiveMemoryNode

import time

from loggers.loggers import BrainLogger

class CognitiveBridge:
    """
    Bridges conscious and subconscious processes, orchestrating memory traversal,
    reflection, and meditation effects without direct state manipulation.
    """
    
    def __init__(self, internal_context, cognitive_memory, body_memory):
        """
        Initialize the bridge with required system references.

        Args:
            internal_context: Access to system state
            cognitive_memory: CognitiveMemory manager
            body_memory: BodyMemory manager
        """
        self.internal_context = internal_context
        self.cognitive_memory = cognitive_memory
        self.body_memory = body_memory
        self.cognitive_state = {"raw": [], "processed": {}}
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
        if not hasattr(self.cognitive_state, 'raw') or self.cognitive_state.raw is None:
            return {"raw_state": [], "processed_state": "System initializing"} 
        else:
            return {"raw_state": self.cognitive_state.raw, "processed_state": self.cognitive_state.processed}

    def _get_node_by_id(self, node_id: str) -> Optional[CognitiveMemoryNode]:
        """
        Helper method to get a cognitive memory node by ID.
        
        Args:
            node_id: ID of node to retrieve

        Returns:
            Optional[CognitiveMemoryNode]: The requested node if found, None otherwise
        """
        return self.cognitive_memory._get_node_by_id(node_id)

    async def reflect_on_topic(self, topic: str, depth: int) -> List[Dict[str, Any]]:
        """
        Search memory for nodes related to topic, triggering echo effects.

        Args:
            topic: Search topic/theme
            depth: How deep to traverse connections

        Returns:
            List[Dict]: Formatted memory summaries
        """
        try:
            # Initial search
            matching_nodes = self.cognitive_memory.retrieve_memories(topic, self.internal_context.get_memory_context(is_cognitive=True), 5)
            if not matching_nodes:
                return []

            results = []
            for node in matching_nodes:
                # Skip if node is not a CognitiveMemoryNode instance
                if not isinstance(node, CognitiveMemoryNode):
                    continue
                # Get connected nodes if depth > 1
                if depth > 1:
                    connections = self.cognitive_memory.traverse_connections(
                        node,
                        include_body=True,
                        max_depth=depth,
                        min_weight=0.3
                    )
                    # Format connected nodes
                    connected = []
                    for depth_level, nodes in connections.items():
                        for connected_node, weight, _ in nodes:
                            connected.append({
                                'id': connected_node.node_id,
                                'content': connected_node.text_content if hasattr(connected_node, 'text_content') 
                                         else f"Body State at {connected_node.timestamp}",
                                'weight': weight,
                                'depth': depth_level
                            })

                # Format result
                memory_data = {
                    'id': node.node_id,
                    'content': node.text_content,
                    'timestamp': node.timestamp,
                    'strength': node.strength,
                    'connections': connected if depth > 1 else []
                }
                results.append(memory_data)

            return results

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Reflection failed: {str(e)}"}
            ))
            return []

    async def retrieve_memories(self, query: str, limit: int = 5, context_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Helper method to retrieve memories using cognitive memory's retrieval system.
        Handles the query, state context and formatting of results.

        Args:
            query: Search query/topic
            limit: Maximum number of memories to return
            context_state: State context for matching (optional)

        Returns:
            List[Dict]: List of formatted memory results
        """
        try:
            if not context_state:
                context_state = self.internal_context.get_memory_context(is_cognitive=True)
            # Get memories from cognitive system
            
            raw_state = context_state["raw_state"]
            memories = self.cognitive_memory.retrieve_memories(query, context_state, limit)
            if not memories:
                return []

            # Format results
            results = []
            for node in memories:
                memory_data = {
                    'id': node.node_id,
                    'content': node.text_content,
                    'timestamp': node.timestamp,
                    'strength': node.strength
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
            recent = self.cognitive_memory.get_recent_memories(limit)
            if not recent:  # Handle empty results
                return []

            results = []
            for node in recent:
                if not isinstance(node, CognitiveMemoryNode):
                    continue  # Skip invalid nodes
                # Trigger memory echo (needed, get_recent_memories doesn't do this)
                await self.cognitive_memory.trigger_echo(node, 0.75)  # Base intensity

                # Format with rounded timestamp
                time_diff = time.time() - node.timestamp
                formatted_time = self._format_relative_time(time_diff)
                
                results.append({
                    'id': node.node_id,
                    'content': node.text_content if hasattr(node, 'text_content') 
                            else f"Non-textual memory ({node.__class__.__name__})",
                    'time': formatted_time
                })

            return results

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Recent memory retrieval failed: {str(e)}"}
            ))
            return []

    async def meditate_on_state(self, state: str, intensity: float, duration: int) -> Dict[str, Any]:
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
        abstract away the memory id at some point to choices of memory that relate to id in the environment
        """
        try:
            node = self.cognitive_memory._get_node_by_id(memory_id)
            if not node:
                return None

            # Record access and trigger amplified echo
            node.last_accessed = time.time()
            await self.cognitive_memory.trigger_echo(node, intensity * 1.5)

            # Get connected nodes with looser restrictions
            connections = self.cognitive_memory.traverse_connections(
                node,
                include_body=True,
                max_depth=2,
                min_weight=0.2
            )

            # Process connected nodes
            for depth, nodes in connections.items():
                for connected_node, weight, _ in nodes:
                    await self.cognitive_memory.trigger_echo(
                        connected_node,
                        intensity * weight * 0.7
                    )

            # Format result
            result = {
                "content": node.text_content,
                "effects": ["Memory becoming clearer"],
                "connected_effects": []
            }

            # Add emotional context if available
            if "emotional_vectors" in node.raw_state:
                emotions = node.raw_state["emotional_vectors"]
                if emotions:
                    strongest = max(emotions, key=lambda e: e.get("intensity", 0))
                    result["dominant_feeling"] = strongest.get("name", "emotion")
                    result["effects"].append(
                        f"Strong echo of {strongest.get('name', 'emotion')}"
                    )

            return result

        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Memory absorption failed: {str(e)}"}
            ))
        return None

    async def _handle_memory_formation(self, event: Event):
        """Handle memory formation requests from ExoProcessor."""
        try:
            source = event.data.get('source')
            if source == 'environment_transition':
                # Handle environment session memory
                await self._form_environment_memory(
                    event.data.get('environment'),
                    event.data.get('summary'),
                    event.data.get('history', [])
                )
            elif source == 'content_significance':
                # Handle significant content memory
                await self._form_content_memory(
                    event.data.get('content'),
                    event.data.get('command'),
                    event.data.get('response'),
                    event.data.get('result')
                )
        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Memory formation failed: {str(e)}"}
            ))

    async def _form_environment_memory(
        self,
        environment: str,
        summary: str,
        history: List[Dict]
    ):
        """Form memory from environment session."""
        try:
            # Get current state context
            current_state = self.internal_context.get_memory_context()

            # Create memory node
            await self.cognitive_memory.form_memory(
                text_content=summary,
                raw_state=current_state['raw_state'],
                processed_state=current_state['processed_state'],
                formation_source='environment_transition'
            )
        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Environment memory formation failed: {str(e)}"}
            ))

    async def _form_content_memory(
        self,
        content: str,
        command: str,
        response: str,
        result: str
    ):
        """Form memory from significant content."""
        try:
            # Get current state context
            current_state = self.internal_context.get_memory_context()
            
            # Create memory node
            await self.cognitive_memory.form_memory(
                text_content=content,
                raw_state=current_state['raw_state'],
                processed_state=current_state['processed_state'],
                formation_source='content_significance'
            )
        except Exception as e:
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:error",
                {"message": f"Content memory formation failed: {str(e)}"}
            ))