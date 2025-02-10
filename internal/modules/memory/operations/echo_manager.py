"""
echo/manager.py

Manages memory echo effects – the mechanism through which retrieved memories
resonate through both cognitive and body networks. Echoes both reinforce 
memory relevance AND propagate the remembered emotional/somatic state through
the system.

Key capabilities:
- Evaluates echo potential using retrieval metrics
- Propagates somatic/emotional effects via events
- Strengthens memories based on echo resonance
- Handles ghost state evaluation during echo
- Cross-network propagation (cognitive <-> body)
"""

import time
from typing import Optional, Dict, Any, List, Tuple

from event_dispatcher import Event, global_event_dispatcher
from loggers.loggers import MemoryLogger

from ..nodes.cognitive_node import CognitiveMemoryNode 
from ..networks.body_network import BodyMemoryNetwork
from ..networks.cognitive_network import CognitiveMemoryNetwork
from ..metrics.orchestrator import RetrievalMetricsOrchestrator


class EchoManager:
    """
    Coordinates echo evaluation and propagation across memory networks.
    Echoes serve dual purposes:
      1. Strengthen memories based on retrieval relevance
      2. Propagate remembered states through the system
    """
    
    def __init__(
        self,
        cognitive_network: CognitiveMemoryNetwork,
        body_network: BodyMemoryNetwork,
        metrics_orchestrator: RetrievalMetricsOrchestrator
    ):
        self.cognitive_network = cognitive_network
        self.body_network = body_network
        self.metrics_orchestrator = metrics_orchestrator
        self.event_dispatcher = global_event_dispatcher
        self.logger = MemoryLogger()

    async def evaluate_echo(
        self,
        node: CognitiveMemoryNode,
        comparison_state: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        precalculated_metrics: Optional[Dict] = None
    ) -> Tuple[Optional[CognitiveMemoryNode], float, List[Dict]]:
        """
        Evaluate echo potential for a node and its ghosts.
        Uses retrieval metrics to determine how strongly the memory
        should resonate in the current context.
        """
        if not node:
            self.logger.warning("evaluate_echo called with None node")
            return None, 0.0, []

        evaluations = []
        try:
            if not comparison_state:
                comparison_state = await self._get_current_state()
                if not comparison_state:
                    return None, 0.0, []

            # Evaluate the main node
            if precalculated_metrics and not node.ghosted:
                parent_metrics = precalculated_metrics
            else:
                parent_metrics = await self._get_node_metrics(
                    node, comparison_state, query_text, query_embedding
                )

            if parent_metrics and 'final_score' in parent_metrics:
                parent_eval = self._calculate_echo_components(
                    node, parent_metrics, ghost_factor=1.0
                )
                if parent_eval:
                    evaluations.append(parent_eval)

            # Evaluate ghost nodes
            for ghost in getattr(node, "ghost_nodes", []):
                try:
                    ghost_node = await self._build_ghost_node(ghost)
                    ghost_metrics = await self._get_node_metrics(
                        ghost_node, comparison_state, query_text, query_embedding
                    )
                    if ghost_metrics and 'final_score' in ghost_metrics:
                        ghost_eval = self._calculate_echo_components(
                            ghost_node, ghost_metrics, ghost_factor=1.25
                        )
                        if ghost_eval:
                            evaluations.append(ghost_eval)
                except Exception as e:
                    self.logger.error(f"Ghost evaluation failed: {e}")
                    continue

            if not evaluations:
                return None, 0.0, []

            selected = max(evaluations, key=lambda x: x["echo"]["intensity"] * x["relevance"])
            final_intensity = selected["echo"]["intensity"] * selected["relevance"]

            return selected["node"], final_intensity, evaluations

        except Exception as e:
            self.logger.error(f"Echo evaluation failed: {e}")
            return None, 0.0, []

    async def trigger_echo(
        self,
        node: CognitiveMemoryNode,
        intensity: float,
        comparison_state: Optional[Dict[str, Any]] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        precalculated_metrics: Optional[Dict] = None
    ) -> None:
        """
        Activate an echo effect, propagating the memory's emotional/somatic
        state through the system and strengthening relevant memories.
        """
        if not node:
            return

        if not comparison_state or not self._validate_state(comparison_state):
            comparison_state = await self._get_current_state()
            if not comparison_state:
                return

        selected_node, final_intensity, details = await self.evaluate_echo(
            node, comparison_state, query_text, query_embedding, precalculated_metrics
        )
        if not selected_node:
            return

        # Update access time
        node.last_accessed = time.time()

        # Apply echo dampening
        final_intensity = self._apply_echo_dampening(selected_node, final_intensity)

        # Get echo components from the selected evaluation details
        selected_eval = next((e for e in details if e["node"] == selected_node), None)
        if not selected_eval:
            return
        strength_boost = self._calculate_strength_boost(
            selected_eval["echo"]["components"],
            final_intensity
        )

        # Apply strength changes
        await self._apply_strength_changes(node, selected_node, strength_boost)

        # Propagate echo effects to body network
        await self._propagate_body_effects(selected_node, final_intensity, strength_boost)

        # Dispatch echo event with complete state info
        self._dispatch_echo_event(selected_node, final_intensity)

        # Re-evaluate network effects (if applicable)
        await self._handle_network_effects(selected_node)

    def _calculate_echo_components(
        self,
        node: CognitiveMemoryNode,
        metrics: Dict[str, Any],
        ghost_factor: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """Calculate echo components from retrieval metrics."""
        try:
            cm = metrics['component_metrics']
            state_metrics = cm.get('state', {})
            state_alignment = self._calculate_state_alignment(state_metrics)

            semantic_match = (
                cm['semantic'].get('embedding_similarity', 0) * 0.6 +
                cm['semantic'].get('cognitive_patterns', 0) * 0.4
            )
            emotional_resonance = (
                cm['emotional'].get('vector_similarity', 0) * 0.5 +
                cm['emotional'].get('valence_shift', 0) * 0.5
            )

            echo_intensity = (
                semantic_match * 0.3 +
                emotional_resonance * 0.4 +
                state_alignment * 0.3
            ) * ghost_factor

            return {
                "node": node,
                "echo": {
                    "intensity": echo_intensity,
                    "components": {
                        "semantic_match": semantic_match,
                        "emotional_resonance": emotional_resonance,
                        "state_alignment": state_alignment
                    }
                },
                "relevance": metrics['final_score'] * ghost_factor,
                "metrics": metrics,
                "is_ghost": (ghost_factor > 1.0)
            }

        except Exception as e:
            self.logger.error(f"Echo component calculation failed: {e}")
            return None

    async def _propagate_body_effects(
        self,
        node: CognitiveMemoryNode,
        echo_intensity: float,
        strength_boost: float
    ) -> None:
        """
        Propagate echo effects to connected body memory nodes.
        """
        try:
            body_links = await self._get_body_links(node)
            for body_id, link_strength in body_links:
                body_node = await self.body_network.get_node(str(body_id))
                if body_node:
                    body_boost = strength_boost * link_strength * 0.8
                    old_strength = body_node.strength
                    body_node.strength = min(1.0, body_node.strength + body_boost)
                    await self.body_network.update_node(body_node)
                    self.logger.debug(
                        f"Body node {body_id} boosted: {old_strength:.2f} -> {body_node.strength:.2f}"
                    )
                    self._dispatch_body_echo_event(body_node, echo_intensity * link_strength)
        except Exception as e:
            self.logger.error(f"Body effect propagation failed: {e}")

    def _dispatch_echo_event(
        self,
        node: CognitiveMemoryNode,
        intensity: float
    ) -> None:
        """
        Dispatch event containing the echo's emotional/state data.
        """
        echo_data = {
            "intensity": intensity,
            "source_node": node.node_id,
            "timestamp": time.time(),
            "metadata": {
                "emotional": node.raw_state.get("emotional_vectors", []),
                "mood": node.raw_state.get("mood", {}),
                "needs": node.raw_state.get("needs", {}),
                "behavior": node.raw_state.get("behavior", {})
            },
            "processed_state": node.processed_state
        }
        self.event_dispatcher.dispatch_event(Event("memory:echo", echo_data))

    def _dispatch_body_echo_event(
        self,
        body_node: Any,
        intensity: float
    ) -> None:
        """
        Dispatch body-specific echo event.
        """
        body_data = {
            "intensity": intensity,
            "source_node": body_node.node_id,
            "raw_state": body_node.raw_state,
            "timestamp": time.time()
        }
        self.event_dispatcher.dispatch_event(Event("memory:body_echo", body_data))

    async def _handle_network_effects(self, node: CognitiveMemoryNode) -> None:
        """
        Handle network updates after an echo:
        - Re-evaluate ghost state
        - Update significant connections
        - Check for conflicts
        """
        # do this elsewhere, perhaps in a handler in the memory system on the event after awaiting the result?
        return

    async def _get_current_state(self) -> Optional[Dict[str, Any]]:
        """Get current internal state for comparison."""
        try:
            state = await self.cognitive_network.get_memory_context(is_cognitive=True)
            if not state:
                self.logger.error("Failed to get cognitive memory context")
                return None
            return state
        except Exception as e:
            self.logger.error(f"Failed to get current state: {e}")
            return None

    def _validate_state(self, state: Dict[str, Any]) -> bool:
        """Ensure state has required structure."""
        return isinstance(state, dict) and 'raw_state' in state and 'processed_state' in state

    async def _get_node_metrics(
        self,
        node: CognitiveMemoryNode,
        comparison_state: Dict[str, Any],
        query_text: Optional[str],
        query_embedding: Optional[List[float]]
    ) -> Optional[Dict[str, Any]]:
        """Get complete retrieval metrics for node."""
        try:
            from ..metrics.orchestrator import MetricsConfiguration
            metrics_config = MetricsConfiguration()
            metrics_config.detailed_metrics = True
            return await self.metrics_orchestrator.calculate_metrics(
                target_node=node,
                comparison_state=comparison_state,
                query_text=query_text,
                query_embedding=query_embedding,
                override_config=metrics_config
            )
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            return None

    def _calculate_state_alignment(self, state_metrics: Dict[str, Any]) -> float:
        """Calculate state alignment from metrics components."""
        try:
            valid_components = [comp for comp in state_metrics.values() if isinstance(comp, dict) and comp]
            if not valid_components:
                return 0.0
            component_sums = []
            for comp in valid_components:
                numeric_values = [v for v in comp.values() if isinstance(v, (int, float))]
                if numeric_values:
                    component_sums.append(sum(numeric_values) / len(numeric_values))
            return sum(component_sums) / len(component_sums) if component_sums else 0.0
        except Exception as e:
            self.logger.error(f"State alignment calculation failed: {e}")
            return 0.0

    def _apply_echo_dampening(
        self,
        node: CognitiveMemoryNode,
        intensity: float
    ) -> float:
        """
        Apply time-based echo dampening to prevent rapid re-triggering.
        """
        try:
            echo_window = 180  # seconds
            time_since_echo = time.time() - (node.last_echo_time or 0)
            if time_since_echo > echo_window:
                node.echo_dampening = 1.0
            else:
                node.echo_dampening = max(0.1, node.echo_dampening * 0.75)
            return intensity * node.echo_dampening
        except Exception as e:
            self.logger.error(f"Echo dampening failed: {e}")
            return intensity

    def _calculate_strength_boost(
        self,
        echo_components: Dict[str, float],
        echo_intensity: float
    ) -> float:
        """
        Calculate strength boost based on echo components.
        """
        try:
            boost = (
                echo_components.get('semantic_match', 0) * 0.3 +
                echo_components.get('emotional_resonance', 0) * 0.5 +
                echo_components.get('state_alignment', 0) * 0.2
            )
            return boost * echo_intensity * 0.2
        except Exception as e:
            self.logger.error(f"Strength boost calculation failed: {e}")
            return 0.0

    async def _apply_strength_changes(
        self,
        parent: CognitiveMemoryNode,
        selected: CognitiveMemoryNode,
        boost: float
    ) -> None:
        """Apply strength changes from echo to relevant nodes."""
        try:
            if selected != parent:
                # Boost ghost connection strength if applicable
                ghost_idx = next((i for i, g in enumerate(parent.ghost_nodes) if g["node_id"] == selected.node_id), None)
                if ghost_idx is not None:
                    parent.ghost_nodes[ghost_idx]["strength"] = min(1.0, parent.ghost_nodes[ghost_idx]["strength"] + (boost * 1.5))
            else:
                parent.strength = min(1.0, parent.strength + boost)
            await self.cognitive_network.update_node(parent)
        except Exception as e:
            self.logger.error(f"Strength application failed: {e}")

    async def _get_body_links(
        self,
        node: CognitiveMemoryNode
    ) -> List[Tuple[str, float]]:
        """Get connected body nodes and link strengths."""
        try:
            links = await self.cognitive_network.cog_db_manager.get_body_links(int(node.node_id))
            return [(str(body_id), strength) for body_id, strength in links]
        except Exception as e:
            self.logger.error(f"Failed to get body links: {e}")
            return []

    async def _build_ghost_node(self, ghost_info: Dict[str, Any]) -> CognitiveMemoryNode:
        """
        Construct temporary node for ghost evaluation.
        """
        embedding = ghost_info.get('embedding', [])
        if not embedding and ghost_info.get('text_content'):
            try:
                embedding = await self.metrics_orchestrator.embedding_manager.encode(ghost_info['text_content'])
            except Exception:
                embedding = []
        return CognitiveMemoryNode(
            node_id=ghost_info.get('node_id'),
            text_content=ghost_info.get('text_content', ''),
            embedding=embedding,
            raw_state=ghost_info.get('raw_state', {}),
            processed_state=ghost_info.get('processed_state', {}),
            timestamp=ghost_info.get('timestamp', 0),
            strength=ghost_info.get('strength', 0),
            ghosted=True,
            last_accessed=None,
            last_echo_time=None,
            echo_dampening=1.0
        )
