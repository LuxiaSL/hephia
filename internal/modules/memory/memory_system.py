"""
memory_system.py

Top-level orchestrator for the Memory System. Integrates:
  - Cognitive and Body memory networks
  - DB managers and synthesis/merge operations
  - Embedding manager and unified metrics orchestrator
  - Operations managers (EchoManager, GhostManager, MergeManager, SynthesisManager)
  - Event listeners for memory formation and retrieval
  - Periodic maintenance tasks (ghost cycles, network maintenance, connection updates)
"""
from __future__ import annotations
import asyncio
import random
import time
from typing import Optional, Union, List, Tuple, Dict, Any

from internal.modules.memory.db.managers import MemoryDBManager
from internal.modules.memory.embedding_manager import EmbeddingManager
from internal.modules.memory.metrics.orchestrator import RetrievalMetricsOrchestrator, MetricsConfiguration
from internal.modules.memory.networks.body_network import BodyMemoryNetwork
from internal.modules.memory.networks.cognitive_network import CognitiveMemoryNetwork
from internal.modules.memory.operations.echo_manager import EchoManager
from internal.modules.memory.operations.ghost_manager import GhostManager
from internal.modules.memory.operations.merge_manager import MergeManager
from internal.modules.memory.operations.consolidation_manager import ConsolidationManager
from internal.modules.memory.operations.synthesis.manager import SynthesisManager
from internal.modules.memory.nodes.body_node import BodyMemoryNode
from internal.modules.memory.nodes.cognitive_node import CognitiveMemoryNode

from internal.internal_context import InternalContext
from api_clients import APIManager
from event_dispatcher import global_event_dispatcher, Event
from loggers.loggers import MemoryLogger

logger = MemoryLogger


class MemorySystemOrchestrator:
    # Define attributes with their types:
    api_manager: APIManager
    internal_context: InternalContext
    db_path: str
    metrics_config: Optional[MetricsConfiguration]
    update_interval: int
    db_manager: MemoryDBManager
    embedding_manager: EmbeddingManager
    metrics_orchestrator: RetrievalMetricsOrchestrator
    body_network: BodyMemoryNetwork
    cognitive_network: CognitiveMemoryNetwork
    echo_manager: EchoManager
    synthesis_manager: SynthesisManager
    merge_manager: MergeManager
    ghost_manager: GhostManager
    consolidation_manager: ConsolidationManager
    _shutting_down: bool
    _pending_operations: int
    _shutdown_lock: asyncio.Lock
    logger: MemoryLogger

    def __init__(
        self,
        api_manager: APIManager,
        internal_context: InternalContext,
        db_path: str = "data/memory.db",
        metrics_config: Optional[MetricsConfiguration] = None,
        update_interval: int = 1800  # in seconds
    ) -> None:
        """
        Initialize the Memory System Orchestrator with all core components.

        Args:
            db_path: Path to the SQLite database.
            metrics_config: Configuration for retrieval metrics.
            update_interval: Interval for periodic maintenance tasks.
        """
        #  Synchronous assignments only
        self.api_manager = api_manager
        self.internal_context = internal_context
        self.db_path = db_path
        self.metrics_config = metrics_config
        self.update_interval = update_interval
        self._shutting_down = False
        self._pending_operations = 0
        self._shutdown_lock = asyncio.Lock()

        # Placeholders for components initialized asynchronously:
        self.db_manager = None  # type: ignore
        self.embedding_manager = None  # type: ignore
        self.metrics_orchestrator = None  # type: ignore
        self.body_network = None  # type: ignore
        self.cognitive_network = None  # type: ignore
        self.echo_manager = None  # type: ignore
        self.synthesis_manager = None  # type: ignore
        self.merge_manager = None  # type: ignore
        self.ghost_manager = None  # type: ignore
        self.consolidation_manager = None  # type: ignore

        self.logger = MemoryLogger

    @classmethod
    async def create(cls, api_manager: APIManager, internal_context: InternalContext,
                     db_path: str = "data/memory.db",
                     metrics_config: Optional[MetricsConfiguration] = None,
                     update_interval: int = 1800) -> MemorySystemOrchestrator:
        logger.info("Initializing MemorySystemOrchestrator...")
        instance = cls(api_manager, internal_context, db_path, metrics_config, update_interval)

        # Initialize DB Manager (and await its initialization)
        instance.db_manager = MemoryDBManager(db_path=db_path)
        await instance.db_manager.init_database()

        # Initialize the embedding and metrics orchestrator
        instance.embedding_manager = EmbeddingManager(api_manager=api_manager)
        instance.metrics_config = metrics_config or MetricsConfiguration()
        instance.metrics_orchestrator = RetrievalMetricsOrchestrator(
            embedding_manager=instance.embedding_manager,
            config=instance.metrics_config
        )

        # Initialize Memory Networks via their async factories
        instance.body_network = await BodyMemoryNetwork.create(
            db_manager=instance.db_manager.body_manager,
            metrics_orchestrator=instance.metrics_orchestrator
        )
        instance.cognitive_network = await CognitiveMemoryNetwork.create(
            db_manager=instance.db_manager.cognitive_manager,
            metrics_orchestrator=instance.metrics_orchestrator
        )

        # Initialize operations managers
        instance.echo_manager = EchoManager(
            cognitive_network=instance.cognitive_network,
            body_network=instance.body_network,
            metrics_orchestrator=instance.metrics_orchestrator
        )
        instance.synthesis_manager = SynthesisManager(
            cognitive_network=instance.cognitive_network,
            db_manager=instance.db_manager.cognitive_manager,
            relation_manager=instance.db_manager.synthesis_relation_manager,
            metrics_orchestrator=instance.metrics_orchestrator
        )
        instance.merge_manager = MergeManager(
            body_network=instance.body_network,
            cognitive_network=instance.cognitive_network,
            metrics_orchestrator=instance.metrics_orchestrator,
            synthesis_manager=instance.synthesis_manager
        )
        instance.ghost_manager = GhostManager(
            body_network=instance.body_network,
            cognitive_network=instance.cognitive_network,
            ghost_threshold=0.1,
            final_prune_threshold=0.05,
            revive_threshold=0.2
        )
        instance.consolidation_manager = ConsolidationManager(
            merge_manager=instance.merge_manager,
            ghost_manager=instance.ghost_manager
        )

        # Set up event listeners
        instance.setup_event_listeners()

        logger.info("MemorySystemOrchestrator initialized.")
        return instance


    def setup_event_listeners(self) -> None:
        """
        Register event listeners for body memory formation and echo dispatch.
        """
        # Body memory formation from internal state events
        global_event_dispatcher.add_listener(
            "emotion:finished",
            lambda event: asyncio.create_task(self.process_memory_formation(event))
        )
        global_event_dispatcher.add_listener(
            "behavior:changed",
            lambda event: asyncio.create_task(self.process_memory_formation(event))
        )
        global_event_dispatcher.add_listener(
            "mood:changed",
            lambda event: asyncio.create_task(self.process_memory_formation(event))
        )
        global_event_dispatcher.add_listener(
            "need:changed",
            lambda event: asyncio.create_task(self.process_memory_formation(event))
        )

        # Echo requests from retrieval routines
        async def handle_echo_request(event: Event) -> None:
            node = await self.cognitive_network.get_node(event.data["node_id"])
            await self.echo_manager.trigger_echo(
                node=node,
                intensity=event.data["similarity"],
                comparison_state=event.data["given_state"],
                query_text=event.data["query_text"],
                query_embedding=event.data["query_embedding"],
                precalculated_metrics=event.data["precalculated_metrics"]
            )
        global_event_dispatcher.add_listener(
            "memory:echo_requested",
            lambda event: asyncio.create_task(handle_echo_request(event))
        )

        # Conflict resolution from synthesis
        global_event_dispatcher.add_listener(
            "cognitive:memory:conflict_resolved",
            lambda event: asyncio.create_task(self.on_conflict_resolved(event))
        )

        logger.debug("Event listeners registered.")

    async def run_maintenance_cycle(self) -> None:
        """
        Run a single maintenance pass: network maintenance, consolidation,
        ghost cycle, and pruning. Called periodically by the timer coordinator.
        """
        try:
            logger.debug("Running maintenance cycle...")

            # Basic network maintenance first
            await self.body_network.maintain_network()
            await self.cognitive_network.maintain_network()

            # Run consolidation checks on both networks
            await self.consolidation_manager.run_consolidation_cycle(
                "body",
                list(self.body_network.nodes.values())
            )
            await self.consolidation_manager.run_consolidation_cycle(
                "cognitive",
                list(self.cognitive_network.nodes.values())
            )

            # Run ghost cycle after consolidation
            await self.ghost_manager.run_ghost_cycle()

            logger.debug("Maintenance cycle complete.")
        except Exception as e:
            logger.error(f"Maintenance cycle error: {e}")

    # -----------------------------
    # Memory Evaluation
    # -----------------------------
    async def _evaluate_node_against_network(
        self,
        temp_node: Union[CognitiveMemoryNode, BodyMemoryNode],
        context: Dict[str, Any],
        evaluation_purpose: str = "strength",
        sample_size: int = 20,
        metrics_config_override: Optional[MetricsConfiguration] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a temporary node against the existing network using metrics.
        Used for both initial strength calculation and significance evaluation.

        Args:
            temp_node: Temporary node to evaluate
            context: Memory context for evaluation
            evaluation_purpose: "strength" or "significance"
            sample_size: Number of recent nodes to compare against
            metrics_config_override: Optional metrics configuration

        Returns:
            Dict containing component scores and final weighted score
        """
        try:
            if isinstance(temp_node, CognitiveMemoryNode):
                comparison_nodes = await self.get_random_nodes(count=sample_size, network_type="cognitive")
            else:
                comparison_nodes = await self.get_random_nodes(count=sample_size, network_type="body")

            MIN_NODES_FOR_REAL_EVAL = 5
            if len(comparison_nodes) < MIN_NODES_FOR_REAL_EVAL:
                self.logger.debug(
                    f"Fewer than {MIN_NODES_FOR_REAL_EVAL} nodes available, "
                    f"using default score for {evaluation_purpose} evaluation."
                )
                return await self._evaluate_against_synthetic_baseline(
                    temp_node=temp_node,
                    context=context,
                    evaluation_purpose=evaluation_purpose
                )

            if not comparison_nodes:
                return {
                    "final_score": 0.5,
                    "component_scores": {"novelty": 0.5, "emotional_impact": 0.5},
                    "method": "no_comparison_nodes"
                }

            # Configure metrics based on purpose and node type
            if metrics_config_override:
                metrics_config = metrics_config_override
            else:
                from .metrics.orchestrator import MetricComponent
                metrics_config = MetricsConfiguration(
                    enabled_components=[MetricComponent.EMOTIONAL],
                    detailed_metrics=True,
                    include_strength=False
                )
                if hasattr(temp_node, 'text_content') and temp_node.text_content:
                    metrics_config.enabled_components.append(MetricComponent.SEMANTIC)

            # Calculate bidirectional metrics against comparison nodes
            significance_scores = []

            for node in comparison_nodes:
                try:
                    temp_node_state = {
                        'raw_state': temp_node.raw_state,
                        'processed_state': temp_node.processed_state
                    }
                    node_state = {
                        'raw_state': node.raw_state,
                        'processed_state': node.processed_state
                    }

                    forward_metrics = await self.metrics_orchestrator.calculate_metrics(
                        target_node=node,
                        comparison_state=temp_node_state,
                        query_text=getattr(temp_node, 'text_content', ''),
                        query_embedding=getattr(temp_node, 'embedding', []),
                        override_config=metrics_config
                    )

                    backward_metrics = await self.metrics_orchestrator.calculate_metrics(
                        target_node=temp_node,
                        comparison_state=node_state,
                        query_text=getattr(node, 'text_content', ''),
                        query_embedding=getattr(node, 'embedding', []),
                        override_config=metrics_config
                    )

                    if isinstance(forward_metrics, dict) and isinstance(backward_metrics, dict):
                        forward_components = forward_metrics.get('component_metrics', {})
                        backward_components = backward_metrics.get('component_metrics', {})

                        combined_metrics = {
                            'semantic': self._average_metric_components(
                                forward_components.get('semantic', {}),
                                backward_components.get('semantic', {})
                            ),
                            'emotional': self._average_metric_components(
                                forward_components.get('emotional', {}),
                                backward_components.get('emotional', {})
                            ),
                        }
                        significance_scores.append(combined_metrics)

                except Exception as e:
                    self.logger.warning(f"Failed metric comparison in {evaluation_purpose}: {e}")
                    continue

            if not significance_scores:
                return {
                    "final_score": 0.5,
                    "component_scores": {"novelty": 0.5, "emotional_impact": 0.5},
                    "method": "metrics_calculation_failed"
                }

            try:
                novelty = self._calculate_novelty(significance_scores)
            except Exception:
                novelty = 0.5

            try:
                emotional_impact = self._calculate_emotional_impact(significance_scores)
            except Exception:
                emotional_impact = 0.5

            weights = self._get_evaluation_weights(temp_node, evaluation_purpose)

            weighted_score = (
                novelty * weights['novelty'] +
                emotional_impact * weights['emotional']
            )

            final_score = max(0.1, min(1.0, weighted_score))

            self.logger.debug(
                f"{evaluation_purpose.title()} eval: novelty={novelty:.3f}, "
                f"emotional={emotional_impact:.3f}, final={final_score:.3f} "
                f"({len(significance_scores)} comparisons)"
            )

            return {
                "final_score": final_score,
                "component_scores": {
                    "novelty": novelty,
                    "emotional_impact": emotional_impact,
                },
                "method": "sophisticated_metrics",
                "comparison_count": len(significance_scores)
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed for {evaluation_purpose}: {e}")
            return {
                "final_score": 0.5,
                "component_scores": {"novelty": 0.5, "emotional_impact": 0.5},
                "method": "error_fallback",
                "error": str(e)
            }

    async def _evaluate_against_synthetic_baseline(
        self,
        temp_node: Union[CognitiveMemoryNode, BodyMemoryNode],
        context: Dict[str, Any],
        evaluation_purpose: str
    ) -> Dict[str, Any]:
        """
        Returns a default score when the network is too sparse for real comparison.
        Cognitive nodes get a slight novelty boost (most early memories are worth keeping).
        """
        if isinstance(temp_node, CognitiveMemoryNode):
            return {
                "final_score": 0.6,
                "component_scores": {"novelty": 0.6, "emotional_impact": 0.5},
                "method": "sparse_network_default",
            }
        else:
            return {
                "final_score": 0.5,
                "component_scores": {"novelty": 0.5, "emotional_impact": 0.5},
                "method": "sparse_network_default",
            }

    def _get_evaluation_weights(
        self,
        node: Union[CognitiveMemoryNode, BodyMemoryNode],
        evaluation_purpose: str
    ) -> Dict[str, float]:
        """
        Get component weights based on node type and evaluation purpose.
        """
        is_cognitive = isinstance(node, CognitiveMemoryNode)

        if evaluation_purpose == "significance":
            if is_cognitive:
                return {'novelty': 0.85, 'emotional': 0.15}
            else:
                return {'novelty': 0.0, 'emotional': 1.0}
        else:
            # Strength evaluation
            if is_cognitive:
                return {'novelty': 0.8, 'emotional': 0.2}
            else:
                return {'novelty': 0.0, 'emotional': 1.0}

    async def evaluate_memory_significance(
        self,
        generated_content: str,
        context: Dict[str, Any],
        source_type: str = "unknown",
        timeout: float = 10.0
    ) -> float:
        """
        Evaluate memory significance using metrics on generated content.
        Called by CognitiveBridge for neuromorphic significance evaluation.

        Args:
            generated_content: LLM-generated memory content
            context: Current memory/cognitive context
            source_type: Type of memory source
            timeout: Maximum evaluation time

        Returns:
            float: Significance score between 0.0 and 1.0
        """
        try:
            async with asyncio.timeout(timeout):
                embedding = await self.embedding_manager.encode(generated_content)

                temp_node = CognitiveMemoryNode(
                    text_content=generated_content,
                    embedding=embedding,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,
                    formation_source="significance_evaluation",
                    node_id="temp_significance_eval",
                    timestamp=time.time()
                )

                from .metrics.orchestrator import MetricComponent
                significance_metrics_config = MetricsConfiguration(
                    enabled_components=[
                        MetricComponent.SEMANTIC,
                        MetricComponent.EMOTIONAL,
                    ],
                    detailed_metrics=True,
                    include_strength=False,
                    component_weights={
                        MetricComponent.SEMANTIC: 0.65,
                        MetricComponent.EMOTIONAL: 0.35,
                        MetricComponent.TEMPORAL: 0.0,
                        MetricComponent.STRENGTH: 0.0
                    }
                )

                evaluation_result = await self._evaluate_node_against_network(
                    temp_node=temp_node,
                    context=context,
                    evaluation_purpose="significance",
                    sample_size=20,
                    metrics_config_override=significance_metrics_config
                )

                significance_score = evaluation_result["final_score"]

                self.logger.info(
                    f"Significance eval for {source_type}: {significance_score:.3f} "
                    f"(method: {evaluation_result.get('method', 'unknown')}, "
                    f"comparisons: {evaluation_result.get('comparison_count', 0)})"
                )

                return significance_score

        except asyncio.TimeoutError:
            self.logger.warning(f"Significance evaluation timeout for {source_type}")
            return 0.5
        except Exception as e:
            self.logger.error(f"Significance evaluation failed for {source_type}: {e}")
            return 0.5

    # -----------------------------
    # Memory Formation Methods
    # -----------------------------
    async def process_memory_formation(self, event: Event) -> None:
        """
        Process an incoming event (from emotion, behavior, mood, need changes)
        and form a body memory snapshot.
        """
        try:
            event_type = event.event_type.split(':')[0]
            metadata = {
                "source_event": event.event_type,
                "timestamp": time.time()
            }
            if event_type == "emotion":
                emotion = event.data.get("emotion")
                if emotion:
                    metadata.update({
                        "emotion_type": emotion.name,
                        "emotion_intensity": emotion.intensity,
                        "emotion_valence": emotion.valence,
                        "emotion_arousal": emotion.arousal,
                        "emotion_source": emotion.source_type,
                        "emotion_source_data": emotion.source_data
                    })
                    await self.form_body_memory(metadata)
                return
            elif event_type in ["behavior", "mood"]:
                metadata.update({
                    "change_type": event_type,
                    "old_state": event.data.get("old_name"),
                    "new_state": event.data.get("new_name")
                })
                await self.form_body_memory(metadata)
                return
            elif event_type == "need":
                old_val = event.data.get("old_value", 50)
                new_val = event.data.get("new_value", 50)
                if abs(new_val - old_val) >= 5:
                    metadata.update({
                        "need_type": event.data.get("need_type"),
                        "old_value": old_val,
                        "new_value": new_val,
                        "change_magnitude": abs(new_val - old_val)
                    })
                    await self.form_body_memory(metadata)
                return
        except Exception as e:
            logger.error(f"Failed to process memory formation event: {e}")

    async def form_body_memory(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Create a new BodyMemoryNode using current context and provided metadata.
        """
        try:
            memory_context = await self.internal_context.get_memory_context()
            if not memory_context:
                raise ValueError("Failed to get memory context")
            raw_state = memory_context.get("raw_state", {})
            processed_state = memory_context.get("processed_state", {})
            initial_strength = await self._calculate_initial_strength(
                await self.internal_context.get_memory_context(is_cognitive=True)
            )

            logger.debug(f"Forming body memory from {metadata.get('source_event', 'unknown')}")

            current_time = time.time()
            new_node = BodyMemoryNode(
                timestamp=current_time,
                raw_state=raw_state,
                processed_state=processed_state,
                strength=initial_strength,
                node_id=None,
                ghosted=False,
                parent_node_id=None,
                ghost_nodes=None,
                ghost_states=None,
                connections=None,
                last_connection_update=None,
                formation_metadata=metadata
            )
            node_id = await self.body_network.add_node(new_node)
            logger.info(f"Formed new BodyMemoryNode with ID {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Failed to form body memory: {e}")
            return None

    async def form_cognitive_memory(
        self,
        text_content: str,
        current_state: Dict[str, Any],
        formation_source: str = "user_input",
        timeout: float = 45.0
    ) -> Optional[str]:
        """
        Create a new CognitiveMemoryNode with text content (typically from LLM generation),
        state snapshots, and a computed embedding.
        """
        try:
            async with asyncio.timeout(timeout):
                embedding = await self.embedding_manager.encode(text_content)
                initial_strength = await self._calculate_initial_strength(
                    context=current_state,
                    text_content=text_content,
                    embedding=embedding
                )
                current_time = time.time()
                raw_state = current_state.get("raw_state", {})
                processed_state = current_state.get("processed_state", {})
                new_node = CognitiveMemoryNode(
                    text_content=text_content,
                    embedding=embedding,
                    raw_state=raw_state,
                    processed_state=processed_state,
                    strength=initial_strength,
                    formation_source=formation_source,
                    node_id=None,
                    ghosted=False,
                    parent_node_id=None,
                    ghost_nodes=None,
                    ghost_states=None,
                    connections=None,
                    last_connection_update=None,
                    timestamp=current_time
                )

                node_id = await self.cognitive_network.add_node(new_node)
                self.logger.info(f"Formed new CognitiveMemoryNode with ID {node_id}")
                return node_id

        except asyncio.TimeoutError:
            self.logger.error(f"Cognitive memory formation timed out after {timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Failed to form cognitive memory: {e}")
            return None

    async def _calculate_initial_strength(
        self,
        context: Dict[str, Any],
        text_content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate initial memory strength using the shared evaluation infrastructure.
        """
        try:
            current_time = time.time()
            temp_node = None

            if text_content is not None:
                if embedding is None:
                    embedding = await self.embedding_manager.encode(text_content)
                temp_node = CognitiveMemoryNode(
                    text_content=text_content,
                    embedding=embedding,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,
                    formation_source="temporary",
                    node_id="temp",
                    timestamp=current_time
                )
            else:
                temp_node = BodyMemoryNode(
                    timestamp=current_time,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,
                    node_id="temp",
                    formation_metadata=metadata or {}
                )

            evaluation_result = await self._evaluate_node_against_network(
                temp_node=temp_node,
                context=context,
                evaluation_purpose="strength",
                sample_size=20,
                metrics_config_override=None
            )

            return evaluation_result["final_score"]

        except Exception as e:
            self.logger.error(f"Failed to calculate initial strength: {e}")
            return 0.5

    def _average_metric_components(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """
        Average two sets of metric components, handling nested dictionaries recursively.
        """
        result = {}
        all_keys = set(metrics1.keys()) | set(metrics2.keys())

        for key in all_keys:
            val1 = metrics1.get(key, 0.0)
            val2 = metrics2.get(key, 0.0)

            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                result[key] = (val1 + val2) / 2
            elif isinstance(val1, dict) and isinstance(val2, dict):
                result[key] = self._average_metric_components(val1, val2)
            elif isinstance(val1, dict) and not isinstance(val2, dict):
                result[key] = val1.copy() if val1 else {}
            elif isinstance(val2, dict) and not isinstance(val1, dict):
                result[key] = val2.copy() if val2 else {}
            elif hasattr(val1, 'item') and hasattr(val2, 'item'):
                result[key] = (float(val1.item()) + float(val2.item())) / 2
            elif hasattr(val1, 'item') and isinstance(val2, (int, float)):
                result[key] = (float(val1.item()) + val2) / 2
            elif isinstance(val1, (int, float)) and hasattr(val2, 'item'):
                result[key] = (val1 + float(val2.item())) / 2
            else:
                try:
                    result[key] = (float(val1) + float(val2)) / 2
                except (ValueError, TypeError):
                    result[key] = val1 if val1 not in (0.0, 0, None) else val2

        return result

    def _calculate_novelty(self, metrics_list: List[Dict]) -> float:
        """Calculate novelty using semantic discrimination."""
        try:
            semantic_scores = []

            for metrics in metrics_list:
                semantic_metrics = metrics.get('semantic', {})
                if semantic_metrics:
                    embedding_sim = semantic_metrics.get('embedding_similarity', 0.0)
                    text_relevance = semantic_metrics.get('text_relevance', 0.0)
                    semantic_density = semantic_metrics.get('semantic_density', 0.5)
                    semantic_cohesion = semantic_metrics.get('semantic_cohesion', 0.5)

                    novelty_score = (
                        (1.0 - embedding_sim) * 0.35 +
                        semantic_density * 0.40 +
                        (1.0 - text_relevance) * 0.15 +
                        semantic_cohesion * 0.10
                    )
                    semantic_scores.append(novelty_score)

            return max(semantic_scores) if semantic_scores else 0.5
        except Exception as e:
            self.logger.error(f"Failed to calculate novelty: {e}")
            return 0.5

    def _calculate_emotional_impact(self, metrics_list: List[Dict]) -> float:
        """Calculate emotional impact from EmotionalMetricsCalculator metrics."""
        try:
            impact_scores = []

            for metrics in metrics_list:
                emotional_metrics = metrics.get('emotional', {})
                if emotional_metrics:
                    vector_sim = emotional_metrics.get('vector_similarity', 0.0)
                    complexity = emotional_metrics.get('emotional_complexity', 0.0)
                    valence_shift = emotional_metrics.get('valence_shift', 0.0)
                    intensity_delta = emotional_metrics.get('intensity_delta', 0.0)

                    impact_score = (
                        (1.0 - vector_sim) * 0.3 +
                        complexity * 0.2 +
                        valence_shift * 0.25 +
                        intensity_delta * 0.25
                    )
                    impact_scores.append(impact_score)

            return max(impact_scores) if impact_scores else 0.5

        except Exception as e:
            self.logger.error(f"Failed to calculate emotional impact: {e}")
            return 0.5

    # -----------------------------
    # Memory Retrieval Methods
    # -----------------------------
    async def retrieve_cognitive_memories(
        self,
        query: str,
        comparison_state: Dict[str, Any],
        top_k: int = 10,
        threshold: float = 0.0,
        return_details: bool = False,
        dispatch_echo: bool = True
    ) -> Union[List[CognitiveMemoryNode], Tuple[List[CognitiveMemoryNode], List[Dict[str, Any]]]]:
        """
        Retrieve cognitive memories based on a query and a comparison state.
        """
        try:
            query_embedding = await self.embedding_manager.encode(query)
            t_metrics_config = self.metrics_config
            t_metrics_config.detailed_metrics = True
            retrieval_scores = []
            for node in (n for n in self.cognitive_network.nodes.values() if not n.ghosted):
                metrics = await self.metrics_orchestrator.calculate_metrics(
                    target_node=node,
                    comparison_state=comparison_state,
                    query_text=query,
                    query_embedding=query_embedding,
                    override_config=t_metrics_config
                )
                final_score = metrics["final_score"] if isinstance(metrics, dict) else metrics
                retrieval_scores.append((node, final_score, metrics))
            retrieval_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = retrieval_scores[:top_k]

            filtered_results = [res for res in top_results if res[1] >= threshold]
            # Dispatch echo events for filtered top results
            if dispatch_echo:
                for node, similarity, node_metrics in filtered_results:
                    global_event_dispatcher.dispatch_event(Event(
                        "memory:echo_requested",
                        {
                            "node_id": node.node_id,
                            "similarity": similarity,
                            "given_state": comparison_state,
                            "query_text": query,
                            "query_embedding": query_embedding,
                            "precalculated_metrics": node_metrics
                        }
                    ))

            if return_details:
                return ([n for n, _, _ in filtered_results], [m for _, _, m in filtered_results])
            else:
                return [n for n, _, _ in filtered_results]
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return [] if not return_details else ([], [])

    async def get_node_by_id(self, node_id: str) -> Optional[Union[CognitiveMemoryNode, BodyMemoryNode]]:
        """Get a node by ID from either network."""
        node = await self.cognitive_network.get_node(node_id)
        if not node:
            node = await self.body_network.get_node(node_id)
        return node

    async def get_recent_memories(
        self,
        count: int = 5,
        network_type: str = "cognitive",
        include_ghosted: bool = False,
        time_window: Optional[float] = None
    ) -> List[Union[CognitiveMemoryNode, BodyMemoryNode]]:
        """Get most recent memories, optionally within a time window."""
        network = self.cognitive_network if network_type == "cognitive" else self.body_network
        current_time = time.time()
        memories = []
        for node in network.nodes.values():
            if not include_ghosted and node.ghosted:
                continue
            if time_window and (current_time - node.timestamp > time_window):
                continue
            memories.append(node)
        memories.sort(key=lambda x: x.timestamp, reverse=True)
        return memories[:count]

    async def get_random_nodes(
        self,
        count: int,
        network_type: str = "cognitive",
        include_ghosted: bool = False
    ) -> List[Union[CognitiveMemoryNode, BodyMemoryNode]]:
        """Get a random selection of memories from either network."""
        network = self.cognitive_network if network_type == "cognitive" else self.body_network
        nodes = list(network.nodes.values())
        if not include_ghosted:
            nodes = [n for n in nodes if not n.ghosted]
        if len(nodes) < count:
            return nodes
        return random.sample(nodes, count)

    async def query_by_time_window(
        self,
        start_time: float,
        end_time: float,
        network_type: str = "body",
        include_ghosted: bool = False
    ) -> List[Union[BodyMemoryNode, CognitiveMemoryNode]]:
        """Get nodes within a specific time window from either network."""
        if start_time > end_time:
            raise ValueError("Invalid time window")
        try:
            network = self.cognitive_network if network_type == "cognitive" else self.body_network
            nodes = list(network.nodes.values())
            if not include_ghosted:
                nodes = [n for n in nodes if not n.ghosted]
            results = [node for node in nodes if start_time <= node.timestamp <= end_time]
            results.sort(key=lambda x: x.timestamp)
            return results
        except Exception as e:
            logger.error(f"Failed to query time window: {e}")
            raise ValueError(f"Time window query failed: {e}")

    # -----------------------------
    # Conflict Resolution Handler
    # -----------------------------
    async def on_conflict_resolved(self, event: Event) -> None:
        """
        Handle conflict resolution events when LLM synthesis is completed.
        Calculates embedding from synthesis text before passing to synthesis manager.
        """
        try:
            node_a_id = event.data["node_a_id"]
            node_b_id = event.data["node_b_id"]
            synthesis_text = event.data["synthesis_text"]
            resolution_context = event.data.get("resolution_context", {})
            synthesis_embedding = await self.embedding_manager.encode(synthesis_text)
            nodeA = await self.cognitive_network.get_node(node_a_id)
            nodeB = await self.cognitive_network.get_node(node_b_id)
            if not nodeA or not nodeB:
                raise MemoryError("Failed to find nodes for conflict resolution")
            await self.synthesis_manager.handle_conflict_synthesis(
                conflict_data=resolution_context.get("conflicts", {}),
                child=nodeA,
                parent=nodeB,
                synthesis_content=synthesis_text,
                synthesis_embedding=synthesis_embedding,
                additional_strength=0.3
            )
            logger.info(f"Processed conflict resolution between nodes {node_a_id} and {node_b_id}")
        except Exception as e:
            logger.error(f"Failed to handle conflict resolution: {e}")

    # -----------------------------
    # Shutdown
    # -----------------------------
    async def shutdown(self) -> None:
        """Graceful shutdown ensuring all operations complete."""
        async with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True

        logger.info("Shutting down MemorySystemOrchestrator...")

        # Wait for pending operations to complete
        while self._pending_operations > 0:
            logger.debug(f"Waiting for {self._pending_operations} operations to complete...")
            await asyncio.sleep(0.1)

        try:
            await self.cognitive_network.shutdown()
            await self.body_network.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down networks: {e}")

        logger.info("Memory system shutdown complete.")
