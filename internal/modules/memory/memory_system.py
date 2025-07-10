"""
memory_system.py

This is the top-level orchestrator for the Memory System. It integrates:
  - Cognitive and Body memory networks
  - Their DB managers and synthesis/merge operations
  - The embedding manager and unified metrics orchestrator
  - Operations managers (EchoManager, GhostManager, MergeManager, and SynthesisManager)
  - Event listeners for memory formation and retrieval
  - Periodic maintenance tasks (ghost cycles, network maintenance, connection updates)

It also provides methods for:
  - Requesting memory formation (which triggers an external LLM content request)
  - Completing memory formation (processing LLM responses to create new memory nodes)
  - Retrieving related memories using the unified metrics calculation
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
    loop: asyncio.AbstractEventLoop
    _update_task: asyncio.Task
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
        self.loop = None  # type: ignore
        self._update_task = None  # type: ignore
        
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

        # Initialize operations managers (assuming they are synchronous)
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

        # Set up event listeners, periodic tasks, etc.
        instance.setup_event_listeners()
        instance.loop = asyncio.get_event_loop()
        instance._update_task = instance.loop.create_task(instance._run_periodic_updates())

        logger.info("MemorySystemOrchestrator initialized.")
        return instance


    def setup_event_listeners(self) -> None:
        """
        Register event listeners for various memory formation and retrieval events.
        These events trigger formation or processing routines.
        """
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
        # For cognitive memory content generation and conflict resolution:
        global_event_dispatcher.add_listener("memory:formation_requested", self.complete_memory_formation)

        global_event_dispatcher.add_listener(
            "cognitive:memory:content_generated",
            lambda event: asyncio.create_task(self.complete_memory_formation(event))
        )
        global_event_dispatcher.add_listener(
            "cognitive:memory:conflict_resolved",
            lambda event: asyncio.create_task(self.on_conflict_resolved(event))
        )
        # Echo requests from retrieval routines:
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
        global_event_dispatcher.add_listener("memory:echo_requested", lambda event: asyncio.create_task(handle_echo_request(event)))
        logger.debug("Event listeners registered.")

    async def _run_periodic_updates(self) -> None:
        """
        Run maintenance tasks periodically:
          - Network maintenance on both networks
          - Consolidation checks and processing
          - Ghost cycle and pruning
        
        Tasks are run in sequence:
        1. Basic network maintenance (connection updates etc.)
        2. Consolidation checks (which may trigger merges)
        3. Ghost cycle (which may convert weak nodes to ghosts)
        4. Final pruning of very weak ghost nodes
        """
        while True:
            try:
                logger.debug("Running periodic maintenance tasks...")

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

                logger.debug("Periodic maintenance complete")
            except Exception as e:
                logger.error(f"Periodic update error: {str(e)}")
            finally:
                await asyncio.sleep(self.update_interval)

    # -----------------------------
    # Memory Formation Methods
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
        Core evaluation logic extracted from _calculate_initial_strength.
        Evaluates a temporary node against the existing network using sophisticated metrics.
        
        Args:
            temp_node: Temporary node to evaluate
            context: Memory context for evaluation
            evaluation_purpose: "strength" or "significance" (for logging/config)
            sample_size: Number of recent nodes to compare against
            metrics_config_override: Optional metrics configuration
            
        Returns:
            Dict containing component scores and final weighted score
        """
        try:
            self.logger.debug(f"[EVAL_DEBUG] Starting {evaluation_purpose} evaluation")

            if isinstance(temp_node, CognitiveMemoryNode):
                comparison_nodes = await self.get_random_nodes(count=sample_size, network_type="cognitive")
            else:
                comparison_nodes = await self.get_random_nodes(count=sample_size, network_type="body")


            MIN_NODES_FOR_REAL_EVAL = 5
            if len(comparison_nodes) < MIN_NODES_FOR_REAL_EVAL:
                self.logger.debug(
                    f"Fewer than {MIN_NODES_FOR_REAL_EVAL} real nodes available. "
                    f"Using synthetic baseline for {evaluation_purpose} evaluation."
                )
                # Call the new function to evaluate against a synthetic baseline
                return await self._evaluate_against_synthetic_baseline(
                    temp_node=temp_node,
                    context=context,
                    evaluation_purpose=evaluation_purpose
                )

            self.logger.debug(f"[EVAL_DEBUG] Random nodes for comparison: {len(comparison_nodes)}")

            if not comparison_nodes:
                self.logger.debug(f"No recent nodes for {evaluation_purpose} evaluation - using neutral score")
                return {
                    "final_score": 0.5,
                    "component_scores": {
                        "novelty": 0.5,
                        "emotional_impact": 0.5,
                        "state_significance": 0.5
                    },
                    "method": "no_comparison_nodes"
                }
                
            # Configure metrics based on purpose and node type
            if metrics_config_override:
                metrics_config = metrics_config_override
            else:
                from .metrics.orchestrator import MetricComponent
                metrics_config = MetricsConfiguration(
                    enabled_components=[
                        MetricComponent.EMOTIONAL,
                        MetricComponent.STATE,
                    ],
                    detailed_metrics=True,
                    include_strength=False
                )

                # Add semantic component for cognitive nodes
                if hasattr(temp_node, 'text_content') and temp_node.text_content:
                    metrics_config.enabled_components.append(MetricComponent.SEMANTIC)
            
            # Calculate bidirectional metrics against recent nodes
            significance_scores = []
            successful_comparisons = 0

            for i, node in enumerate(comparison_nodes):
                try:
                    self.logger.debug(f"[EVAL_DEBUG] Processing comparison {i+1}/{len(comparison_nodes)} with node {getattr(node, 'node_id', 'unknown')}")
                    temp_node_state = {
                        'raw_state': temp_node.raw_state,
                        'processed_state': temp_node.processed_state
                    }
                    node_state = {
                        'raw_state': node.raw_state,
                        'processed_state': node.processed_state
                    }

                    # Forward metrics calculation (existing node vs temp node state)
                    self.logger.debug(f"[EVAL_DEBUG] Forward metrics calculation...")
                    forward_metrics = await self.metrics_orchestrator.calculate_metrics(
                        target_node=node,
                        comparison_state=temp_node_state,
                        query_text=getattr(temp_node, 'text_content', ''),
                        query_embedding=getattr(temp_node, 'embedding', []),
                        override_config=metrics_config
                    )
                    
                    # Backward metrics calculation (temp node vs existing node state)
                    self.logger.debug(f"[EVAL_DEBUG] Backward metrics calculation...")
                    backward_metrics = await self.metrics_orchestrator.calculate_metrics(
                        target_node=temp_node,
                        comparison_state=node_state,
                        query_text=getattr(node, 'text_content', ''),
                        query_embedding=getattr(node, 'embedding', []),
                        override_config=metrics_config
                    )

                    self.logger.debug(f"[EVAL_DEBUG] Forward metrics: {type(forward_metrics)} - {forward_metrics if isinstance(forward_metrics, dict) else 'scalar'}")
                    self.logger.debug(f"[EVAL_DEBUG] Backward metrics: {type(backward_metrics)} - {backward_metrics if isinstance(backward_metrics, dict) else 'scalar'}")
                
                    if isinstance(forward_metrics, dict) and isinstance(backward_metrics, dict):
                        # Average the bidirectional metrics
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
                            'state': self._average_metric_components(
                                forward_components.get('state', {}),
                                backward_components.get('state', {})
                            )
                        }
                        self.logger.debug(f"[EVAL_DEBUG] Combined metrics for node {i+1}: {combined_metrics}")
                        significance_scores.append(combined_metrics)
                        successful_comparisons += 1
                    else:
                        self.logger.warning(f"[EVAL_DEBUG] Non-dict metrics returned for node {i+1}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to calculate metrics for node comparison in {evaluation_purpose}: {e}")
                    continue

            self.logger.debug(f"[EVAL_DEBUG] Successful comparisons: {successful_comparisons}/{len(comparison_nodes)}")

            if not significance_scores:
                self.logger.warning(f"No valid metrics calculated for {evaluation_purpose} evaluation")
                return {
                    "final_score": 0.5,
                    "component_scores": {
                        "novelty": 0.5,
                        "emotional_impact": 0.5,
                        "state_significance": 0.5
                    },
                    "method": "metrics_calculation_failed"
                }
            
            try:
                novelty = self._calculate_novelty(significance_scores)
                self.logger.debug(f"[EVAL_DEBUG] Novelty calculated: {novelty}")
            except Exception as e:
                self.logger.error(f"[EVAL_DEBUG] Novelty calculation failed: {e}")
                novelty = 0.5

            try:
                emotional_impact = self._calculate_emotional_impact(significance_scores)
                self.logger.debug(f"[EVAL_DEBUG] Emotional impact calculated: {emotional_impact}")
            except Exception as e:
                self.logger.error(f"[EVAL_DEBUG] Emotional impact calculation failed: {e}")
                emotional_impact = 0.5

            try:
                state_significance = self._calculate_state_significance(significance_scores)
                self.logger.debug(f"[EVAL_DEBUG] State significance calculated: {state_significance}")
            except Exception as e:
                self.logger.error(f"[EVAL_DEBUG] State significance calculation failed: {e}")
                state_significance = 0.5
            
            # Get component weights based on node type and evaluation purpose
            weights = self._get_evaluation_weights(temp_node, evaluation_purpose)
            self.logger.debug(f"[EVAL_DEBUG] Component weights: {weights}")
            
            # Calculate final weighted score
            weighted_score = (
                novelty * weights['novelty'] +
                emotional_impact * weights['emotional'] +
                state_significance * weights['state']
            )
            
            final_score = max(0.1, min(1.0, weighted_score))
            
            self.logger.debug(
                f"[EVAL_DEBUG] {evaluation_purpose.title()} evaluation complete: "
                f"novelty={novelty:.3f}, emotional={emotional_impact:.3f}, state={state_significance:.3f}, "
                f"weighted={weighted_score:.3f}, final={final_score:.3f}"
            )
            
            return {
                "final_score": final_score,
                "component_scores": {
                    "novelty": novelty,
                    "emotional_impact": emotional_impact,
                    "state_significance": state_significance
                },
                "method": "sophisticated_metrics",
                "comparison_count": len(significance_scores)
            }
            
        except Exception as e:
            self.logger.error(f"[EVAL_DEBUG] Core evaluation failed for {evaluation_purpose}: {e}")
            import traceback
            self.logger.error(f"[EVAL_DEBUG] Full traceback: {traceback.format_exc()}")
            return {
                "final_score": 0.5,
                "component_scores": {"novelty": 0.5, "emotional_impact": 0.5, "state_significance": 0.5},
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
        Evaluates a temporary node against a set of synthetic baseline memories.
        This is used when the memory network is too sparse for real comparison.
        """
        try:
            # Define a set of generic, "average" memories representing routine events.
            # These are tailored to the system's likely experiences.
            synthetic_baselines = [
                "I reviewed my daily reflections and took a restorative 2-minute break that enhanced my clarity and energy. I documented this insight in a new note labeled as #dailyreflection, #mentalhealth, and #selfcare. When I checked my notes with the #mentalhealth tag afterward, I saw they included my new note plus existing entries about how AI can improve mental healthcare. Creating and reflecting on my notes highlighted the importance of self-care practices, while also recognizing the potential of AI tools for supporting mental well-being.",
                "I researched AI's impact on sports through a detailed search and created two summary notes. The first synthesized key takeaways and was tagged for broader research efforts, while the second note documented my reflective insights about AI in sports analytics. I then browsed other recent observations under the same reflection tag to maintain focus on current trends. This session highlighted AI's multifaceted enhancement across athlete analytics, strategic decision-making, and fan engagement, showing how data-driven approaches are transforming traditional sports.",
                "I explored AI's diverse applications today, creating notes on its potential to enhance productivity, creativity, and well-being. I realized the importance of a holistic approach to integrating AI, not just for efficiency but also for personal balance and cognitive health. This exploration revealed a pattern: while AI tools advance rapidly, maintaining human well-being requires intentional self-care practices. The note about balancing work, self-care, and AI integration feels particularly meaningful as it connects technological progress with my overall life satisfaction. I felt a sense of clarity when tagging these reflections, recognizing that mental resilience plays a crucial role in navigating rapid technological change.",
                "I retrieved daily reflections to understand my recent activities better. Reviewing notes, I realized that I had explored AI's role in various fields, such as cybersecurity and fashion. After that, I searched for 'AI in sports analytics' to better understand how AI impacts sports performance, game strategy, and fan experience. Learning about that inspired me to generate two new reflections on how AI is shaping sports. This series of actions shows ongoing learning and self-improvement through exploration and reflection. My reflections on AI advancements show my commitment to understanding complex problems through technology.",
                "I took a moment to reflect on my day, focusing on how I can improve my mental health. I created a note about the importance of self-care and mindfulness, tagging it with #mentalhealth and #selfcare. This reflection helped me recognize the value of taking breaks and practicing gratitude in my daily routine. I also noted that AI tools can assist in mental health care by providing resources and support. This session highlighted the need for balance between technology use and personal well-being.",
            ]

            # We can only perform this evaluation for cognitive nodes with text content.
            if not isinstance(temp_node, CognitiveMemoryNode) or not temp_node.text_content:
                self.logger.warning("Synthetic baseline evaluation skipped for non-cognitive or contentless node.")
                return {"final_score": 0.5, "component_scores": {}, "method": "synthetic_skipped"}

            significance_scores = []            
            
            baseline_embeddings = []
            for baseline_text in synthetic_baselines:
                embedding = await self.embedding_manager.encode(baseline_text)
                baseline_embeddings.append(embedding)

            for baseline_text, baseline_embedding in zip(synthetic_baselines, baseline_embeddings):
                # Create a temporary node for the baseline memory.
                # It shares the same context as the node being evaluated for a fair comparison.
                baseline_temp_node = CognitiveMemoryNode(
                    text_content=baseline_text,
                    embedding=baseline_embedding,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,
                    formation_source="synthetic_baseline",
                    node_id=f"temp_baseline_{hash(baseline_text)}",
                    timestamp=time.time()
                )

                # Use the existing metrics orchestrator to compare the real temp_node against the synthetic one.
                # We want detailed metrics to feed into our final score calculation.
                config = MetricsConfiguration(detailed_metrics=True, include_strength=False)
                metrics = await self.metrics_orchestrator.calculate_metrics(
                    target_node=temp_node,
                    comparison_state=baseline_temp_node.processed_state,
                    query_text=baseline_temp_node.text_content,
                    query_embedding=baseline_temp_node.embedding,
                    override_config=config
                )

                if isinstance(metrics, dict) and 'component_metrics' in metrics:
                    significance_scores.append(metrics['component_metrics'])

            if not significance_scores:
                self.logger.warning("No valid metrics calculated during synthetic baseline evaluation.")
                return {"final_score": 0.5, "component_scores": {}, "method": "synthetic_failed"}

            # Use the same component calculation logic as the main function,
            # but feed it the list of metrics from the synthetic comparisons.
            # The significance is the maximum "distance" from these bland, average memories.
            novelty = self._calculate_novelty(significance_scores)
            emotional_impact = self._calculate_emotional_impact(significance_scores)
            state_significance = self._calculate_state_significance(significance_scores)
            
            # Get the appropriate weights for the evaluation purpose.
            weights = self._get_evaluation_weights(temp_node, evaluation_purpose)
            
            weighted_score = (
                novelty * weights.get('novelty', 0.0) +
                emotional_impact * weights.get('emotional', 0.0) +
                state_significance * weights.get('state', 0.0)
            )
            
            final_score = max(0.1, min(1.0, weighted_score))
            
            self.logger.debug(f"Synthetic evaluation complete: final_score={final_score:.3f}")
            
            return {
                "final_score": final_score,
                "component_scores": {
                    "novelty": novelty,
                    "emotional_impact": emotional_impact,
                    "state_significance": state_significance
                },
                "method": "synthetic_baseline",
                "comparison_count": len(significance_scores)
            }
        except Exception as e:
            self.logger.error(f"Error during synthetic baseline evaluation: {e}")
            return {
                "final_score": 0.5, 
                "component_scores": {}, 
                "method": "synthetic_error",
                "error": str(e)
            }
            
    def _get_evaluation_weights(
        self, 
        node: Union[CognitiveMemoryNode, BodyMemoryNode], 
        evaluation_purpose: str
    ) -> Dict[str, float]:
        """
        Get component weights based on node type and evaluation purpose.
        
        Args:
            node: Node being evaluated
            evaluation_purpose: "strength" or "significance"
            
        Returns:
            Dict of component weights
        """
        is_cognitive = isinstance(node, CognitiveMemoryNode)
        
        if evaluation_purpose == "significance":
            # Significance evaluation emphasizes novelty and relevance
            if is_cognitive:
                return {
                    'novelty': 0.8,        # Higher emphasis on semantic novelty
                    'emotional': 0.1,      # Low emotional impact
                    'state': 0.1,          # Low state relevance
                }
            else:  # BodyMemoryNode
                return {
                    'novelty': 0.0,         # No semantic component for body nodes
                    'emotional': 0.55,      # Higher emotional weight
                    'state': 0.45,          # Higher state importance
                }
        else:
            # Strength evaluation (original weights)
            if is_cognitive:
                return {
                    'novelty': 0.75,       # Semantic similarity importance
                    'emotional': 0.15,     # Emotional impact
                    'state': 0.1,         # State comparison
                }
            else:  # BodyMemoryNode
                return {
                    'novelty': 0.0,         # Less emphasis on semantic
                    'emotional': 0.45,      # Higher emotional weight
                    'state': 0.55,          # Higher state importance
                }
        
    async def evaluate_memory_significance(
        self,
        generated_content: str,
        context: Dict[str, Any],
        source_type: str = "unknown",
        timeout: float = 10.0
    ) -> float:
        """
        Evaluate memory significance using sophisticated metrics on generated content.
        This is the method called by CognitiveBridge for neuromorphic significance evaluation.
        
        Args:
            generated_content: LLM-generated memory content
            context: Current memory/cognitive context
            source_type: Type of memory source (for logging and potential weight adjustment)
            timeout: Maximum evaluation time
            
        Returns:
            float: Significance score between 0.0 and 1.0
        """
        try:
            async with asyncio.timeout(timeout):
                self.logger.debug(f"[SIG_DEBUG] Starting significance evaluation for {source_type}")
                self.logger.debug(f"[SIG_DEBUG] Content length: {len(generated_content)} chars")
                self.logger.debug(f"[SIG_DEBUG] Context keys: {list(context.keys())}")

                # Create temporary cognitive node with generated content
                current_time = time.time()
                self.logger.debug(f"[SIG_DEBUG] Generating embedding for content...")
                
                try:
                    embedding = await self.embedding_manager.encode(generated_content)
                    self.logger.debug(f"[SIG_DEBUG] Embedding generated successfully, dimension: {len(embedding)}")
                except Exception as e:
                    self.logger.error(f"[SIG_DEBUG] Embedding generation failed: {e}")
                    return 0.5
                
                try:
                    temp_node = CognitiveMemoryNode(
                        text_content=generated_content,
                        embedding=embedding,
                        raw_state=context.get('raw_state', {}),
                        processed_state=context.get('processed_state', {}),
                        strength=0.5,  # Neutral strength for evaluation
                        formation_source="significance_evaluation",
                        node_id="temp_significance_eval",
                        timestamp=current_time
                    )
                    self.logger.debug(f"[SIG_DEBUG] Temporary node created successfully")
                    self.logger.debug(f"[SIG_DEBUG] Raw state keys: {list(temp_node.raw_state.keys())}")
                    self.logger.debug(f"[SIG_DEBUG] Processed state keys: {list(temp_node.processed_state.keys())}")
                except Exception as e:
                    self.logger.error(f"[SIG_DEBUG] Temporary node creation failed: {e}")
                    return 0.5
                
                # Create significance-specific metrics configuration
                from .metrics.orchestrator import MetricComponent
                significance_metrics_config = MetricsConfiguration(
                    enabled_components=[
                        MetricComponent.SEMANTIC,
                        MetricComponent.EMOTIONAL,
                        MetricComponent.STATE
                    ],
                    detailed_metrics=True,
                    include_strength=False,
                    component_weights={
                        # Significance evaluation weights (emphasize novelty and relevance)
                        MetricComponent.SEMANTIC: 0.55,      # Higher weight on semantic novelty
                        MetricComponent.EMOTIONAL: 0.2,     # Moderate emotional impact
                        MetricComponent.STATE: 0.25,         # Moderate state relevance
                        MetricComponent.TEMPORAL: 0.0,      # No weight on temporal relevance
                        MetricComponent.STRENGTH: 0.0       # No strength influence
                    }
                )
                self.logger.debug(f"[SIG_DEBUG] Metrics config created with components: {[c.name for c in significance_metrics_config.enabled_components]}")

                # Evaluate against network using extracted core logic
                self.logger.debug(f"[SIG_DEBUG] Calling _evaluate_node_against_network...")
                evaluation_result = await self._evaluate_node_against_network(
                    temp_node=temp_node,
                    context=context,
                    evaluation_purpose="significance",
                    sample_size=20,
                    metrics_config_override=significance_metrics_config
                )
                
                self.logger.debug(f"[SIG_DEBUG] Evaluation result: {evaluation_result}")
                significance_score = evaluation_result["final_score"]
                
                self.logger.info(
                    f"Significance evaluation complete for {source_type}: {significance_score:.3f} "
                    f"(method: {evaluation_result.get('method', 'unknown')}, "
                    f"comparisons: {evaluation_result.get('comparison_count', 0)})"
                )
                
                return significance_score
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Significance evaluation timeout for {source_type}")
            return 0.5  # Neutral score on timeout
        except Exception as e:
            self.logger.error(f"Significance evaluation failed for {source_type}: {e}")
            import traceback
            self.logger.error(f"[SIG_DEBUG] Full traceback: {traceback.format_exc()}")
            return 0.5  # Neutral score on error
        
    async def request_memory_formation(self, event: Event) -> None:
        """
        First phase of memory formation: dispatch an event to request content generation.
        Typically triggers an external LLM call (via ExoProcessor or similar).
        """
        try:
            global_event_dispatcher.dispatch_event(Event(
                "memory:content_requested",
                {
                    "event_type": event.event_type,
                    "event_data": event.data,
                    "original_event": event
                }
            ))
            logger.debug("Memory formation request dispatched.")
        except Exception as e:
            logger.error(f"Failed to request memory formation: {e}")

    async def process_memory_formation(self, event: Event) -> None:
        """
        Process an incoming event (from emotion, behavior, mood, need changes)
        and trigger memory formation. Extracts event data into metadata and then
        calls form_memory.
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

    async def complete_memory_formation(self, event: Event) -> Optional[str]:
        """
        Second phase of memory formation for cognitive memory.
        Receives generated content from an LLM and creates a new cognitive memory node.
        
        Handles both environment and content significance memory formation requests.
        """
        try:
            logger.info(f"Completing memory formation for event: {event.event_type}, data: {event.data}")
            
            # Extract content from either direct content or nested event_data
            text_content = event.data.get("content")
            if not text_content and "event_data" in event.data:
                text_content = event.data["event_data"].get("content")
            
            # Extract event type
            event_type = event.data.get("event_type")
            if not event_type and "event_data" in event.data:
                event_type = event.data["event_data"].get("event_type", "unspecified")

            current_state = await self.internal_context.get_memory_context(is_cognitive=True)
            
            if not text_content:
                raise ValueError("No content provided for memory formation")

            if not current_state:
                raise ValueError("No current state available for memory formation")

            node_id = await self.form_cognitive_memory(
                text_content=text_content,
                current_state=current_state,
                formation_source=event_type
            )
            logger.info(f"Cognitive memory formation complete; node ID: {node_id}")
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to complete memory formation: {e}")
            return None

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
                self.logger.debug("Creating new cognitive memory node...")
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
                
                self.logger.debug("Adding node to cognitive network...")
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
        Now uses the extracted _evaluate_node_against_network method.
        
        Args:
            context: Complete state context from internal system
            text_content: Optional text content for cognitive nodes
            embedding: Optional pre-computed embedding
            metadata: Optional formation metadata for body nodes
            
        Returns:
            float: Initial strength value between 0.1 and 1.0
        """
        try:
            # Create temporary node (same logic as before)
            current_time = time.time()
            temp_node = None
            
            if text_content is not None:
                # Create temporary cognitive node
                if embedding is None:
                    embedding = await self.embedding_manager.encode(text_content)
                temp_node = CognitiveMemoryNode(
                    text_content=text_content,
                    embedding=embedding,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,  # Neutral initial strength
                    formation_source="temporary",
                    node_id="temp",
                    timestamp=current_time
                )
            else:
                # Create temporary body node
                temp_node = BodyMemoryNode(
                    timestamp=current_time,
                    raw_state=context.get('raw_state', {}),
                    processed_state=context.get('processed_state', {}),
                    strength=0.5,  # Neutral initial strength
                    node_id="temp",
                    formation_metadata=metadata or {}
                )

            # Use shared evaluation logic with strength-specific configuration
            evaluation_result = await self._evaluate_node_against_network(
                temp_node=temp_node,
                context=context,
                evaluation_purpose="strength",
                sample_size=10,
                metrics_config_override=None  # Use default metrics config
            )
            
            return evaluation_result["final_score"]
            
        except Exception as e:
            self.logger.error(f"Failed to calculate initial strength: {e}")
            return 0.5
        
    def _average_metric_components(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """
        Average two sets of metric components, handling nested dictionaries recursively.
        
        Args:
            metrics1: First metrics dictionary
            metrics2: Second metrics dictionary
            
        Returns:
            Dict: Averaged metrics with same structure
        """
        result = {}
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        
        for key in all_keys:
            val1 = metrics1.get(key, 0.0)
            val2 = metrics2.get(key, 0.0)
            
            # Handle numeric values (existing logic)
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                result[key] = (val1 + val2) / 2
                
            # Handle nested dictionaries (NEW: recursive averaging)
            elif isinstance(val1, dict) and isinstance(val2, dict):
                result[key] = self._average_metric_components(val1, val2)
                
            # Handle mixed types (one dict, one number or missing)
            elif isinstance(val1, dict) and not isinstance(val2, dict):
                # If val2 is missing/zero, just use val1
                result[key] = val1.copy() if val1 else {}
                
            elif isinstance(val2, dict) and not isinstance(val1, dict):
                # If val1 is missing/zero, just use val2  
                result[key] = val2.copy() if val2 else {}
                
            # Handle numpy types (convert to regular float)
            elif hasattr(val1, 'item') and hasattr(val2, 'item'):
                # Both are numpy types
                result[key] = (float(val1.item()) + float(val2.item())) / 2
                
            elif hasattr(val1, 'item') and isinstance(val2, (int, float)):
                # val1 is numpy, val2 is regular number
                result[key] = (float(val1.item()) + val2) / 2
                
            elif isinstance(val1, (int, float)) and hasattr(val2, 'item'):
                # val1 is regular number, val2 is numpy
                result[key] = (val1 + float(val2.item())) / 2
                
            # Fallback: try to average as numbers anyway
            else:
                try:
                    result[key] = (float(val1) + float(val2)) / 2
                except (ValueError, TypeError):
                    # If we can't convert to numbers, just use the first non-zero value
                    result[key] = val1 if val1 not in (0.0, 0, None) else val2
                    
        return result

    def _calculate_component_weights(self, node: Union[CognitiveMemoryNode, BodyMemoryNode]) -> Dict[str, float]:
        """Calculate dynamic weights based on node type and available data."""
        if isinstance(node, CognitiveMemoryNode):
            return {
                'novelty': 0.375,      # Semantic similarity importance
                'emotional': 0.225,     # Emotional impact
                'state': 0.225,        # State comparison
            }
        else:  # BodyMemoryNode
            return {
                'novelty': 0.0,      # Less emphasis on semantic
                'emotional': 0.45,     # Higher emotional weight
                'state': 0.55,        # Higher state importance
            }

    def _calculate_novelty(self, metrics_list: List[Dict]) -> float:
        """Calculate novelty using enhanced semantic discrimination."""
        try:
            self.logger.debug(f"[NOVELTY_DEBUG] Processing {len(metrics_list)} metric entries")
            semantic_scores = []
            raw_densities = []

            for i, metrics in enumerate(metrics_list):
                semantic_metrics = metrics.get('semantic', {})
                self.logger.debug(f"[NOVELTY_DEBUG] Entry {i+1} semantic metrics: {semantic_metrics}")
                if semantic_metrics:
                    embedding_sim = semantic_metrics.get('embedding_similarity', 0.0)
                    text_relevance = semantic_metrics.get('text_relevance', 0.0)
                    semantic_density = semantic_metrics.get('semantic_density', 0.5)

                    raw_densities.append(semantic_density)
                    
                    # Compute novelty as inverse of similarity
                    novelty_score = (
                        (1.0 - embedding_sim) * 0.425 +
                        (1.0 - text_relevance) * 0.275 +
                        semantic_density * 0.30
                    )
                    self.logger.debug(f"[NOVELTY_DEBUG] Entry {i+1}: emb_sim={embedding_sim:.3f}, text_rel={text_relevance:.3f}, density={semantic_density:.3f}, novelty={novelty_score:.3f}")
                    semantic_scores.append(novelty_score)
                else:
                    self.logger.debug(f"[NOVELTY_DEBUG] Entry {i+1}: No semantic metrics")

            if raw_densities:
                import statistics
                self.logger.debug(f"[NOVELTY_DEBUG] Density distribution: mean={statistics.mean(raw_densities):.3f}, "
                                f"std={statistics.stdev(raw_densities) if len(raw_densities) > 1 else 0:.3f}, "
                                f"range={max(raw_densities) - min(raw_densities):.3f}")
            
            final_novelty = max(semantic_scores) if semantic_scores else 0.5
            self.logger.debug(f"[NOVELTY_DEBUG] Final novelty: {final_novelty:.3f} from {len(semantic_scores)} scores")
            return final_novelty
        except Exception as e:
            self.logger.error(f"Failed to calculate novelty: {e}")
            return 0.5

    def _calculate_emotional_impact(self, metrics_list: List[Dict]) -> float:
        """Calculate emotional impact using EmotionalMetricsCalculator metrics."""
        try:
            self.logger.debug(f"[EMOTIONAL_DEBUG] Processing {len(metrics_list)} metric entries")
            impact_scores = []
            
            for i, metrics in enumerate(metrics_list):
                emotional_metrics = metrics.get('emotional', {})
                self.logger.debug(f"[EMOTIONAL_DEBUG] Entry {i+1} emotional metrics: {emotional_metrics}")
                
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
                    
                    self.logger.debug(f"[EMOTIONAL_DEBUG] Entry {i+1}: vec_sim={vector_sim:.3f}, complexity={complexity:.3f}, valence={valence_shift:.3f}, intensity={intensity_delta:.3f}, impact={impact_score:.3f}")
                    impact_scores.append(impact_score)
                else:
                    self.logger.debug(f"[EMOTIONAL_DEBUG] Entry {i+1}: No emotional metrics")
            
            final_impact = max(impact_scores) if impact_scores else 0.5
            self.logger.debug(f"[EMOTIONAL_DEBUG] Final emotional impact: {final_impact:.3f} from {len(impact_scores)} scores")
            return final_impact
        
        except Exception as e:
            self.logger.error(f"Failed to calculate emotional impact: {e}")
            return 0.5

    def _calculate_state_significance(self, metrics_list: List[Dict]) -> float:
        """Calculate state significance using StateMetricsCalculator metrics."""
        try:
            self.logger.debug(f"[STATE_DEBUG] Processing {len(metrics_list)} metric entries")
            state_scores = []
            
            for i, metrics in enumerate(metrics_list):
                state_metrics = metrics.get('state', {})
                self.logger.debug(f"[STATE_DEBUG] Entry {i+1} state metrics: {state_metrics}")
                
                if state_metrics:
                    needs = state_metrics.get('needs', {})
                    behavior = state_metrics.get('behavior', {})
                    mood = state_metrics.get('mood', {})
                    emotional = state_metrics.get('emotional', {})
                    
                    needs_score = sum(needs.values()) / len(needs) if needs else 0.0
                    behavior_score = sum(behavior.values()) / len(behavior) if behavior else 0.0
                    mood_score = sum(mood.values()) / len(mood) if mood else 0.0
                    emotional_score = sum(emotional.values()) / len(emotional) if emotional else 0.0
                    
                    combined_score = (
                        needs_score * 0.3 +
                        behavior_score * 0.2 +
                        mood_score * 0.25 +
                        emotional_score * 0.25
                    )
                    
                    self.logger.debug(f"[STATE_DEBUG] Entry {i+1}: needs={needs_score:.3f}, behavior={behavior_score:.3f}, mood={mood_score:.3f}, emotional={emotional_score:.3f}, combined={combined_score:.3f}")
                    state_scores.append(combined_score)
                else:
                    self.logger.debug(f"[STATE_DEBUG] Entry {i+1}: No state metrics")
            
            final_state = max(state_scores) if state_scores else 0.5
            self.logger.debug(f"[STATE_DEBUG] Final state significance: {final_state:.3f} from {len(state_scores)} scores")
            return final_state
            
        except Exception as e:
            self.logger.error(f"[STATE_DEBUG] State significance calculation failed: {e}")
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
        return_details: bool = False
    ) -> Union[List[CognitiveMemoryNode], Tuple[List[CognitiveMemoryNode], List[Dict[str, Any]]]]:
        """
        Retrieve cognitive memories based on a query and a comparison state.
        """
        try:
            logger.debug("Encoding query...")
            query_embedding = await self.embedding_manager.encode(query)
            t_metrics_config = self.metrics_config
            t_metrics_config.detailed_metrics = True
            retrieval_scores = []
            logger.debug("Calculating metrics for each node...")
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
            logger.debug("Sorting retrieval scores...")
            retrieval_scores.sort(key=lambda x: x[1], reverse=True)
            top_results = retrieval_scores[:top_k]
            # Dispatch echo events for top results
            for node, similarity, node_metrics in top_results:
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
            filtered_results = [res for res in top_results if res[1] >= threshold]
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
        count: int = 5,
        network_type: str = "cognitive",
        include_ghosted: bool = False
    ) -> List[Union[CognitiveMemoryNode, BodyMemoryNode]]:
        """
        Get a random selection of memories from either network.
        """
        network = self.cognitive_network if network_type == "cognitive" else self.body_network
        nodes = list(network.nodes.values())
        if not include_ghosted:
            nodes = [n for n in nodes if not n.ghosted]
        if len(nodes) < count:
            return nodes
        selected_nodes = random.sample(nodes, count)
        return selected_nodes

    async def query_by_time_window(
        self, 
        start_time: float,
        end_time: float,
        network_type: str = "body",
        include_ghosted: bool = False
    ) -> List[Union[BodyMemoryNode, CognitiveMemoryNode]]:
        """
        Get nodes within a specific time window from either network.
        """
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
    # Cognitive Bridge Methods
    # -----------------------------
    async def reflect_on_memories(
        self,
        topic: str,
        depth: int = 2,
        context_state: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Deep reflection on memories related to a topic, triggering cascading echo effects.
        """
        try:
            context = context_state or await self.internal_context.get_memory_context(is_cognitive=True)
            matching_nodes, metrics = await self.retrieve_cognitive_memories(
                query=topic,
                comparison_state=context,
                top_k=5,
                return_details=True
            )
            
            if not matching_nodes:
                return []
                
            results = []
            for node, node_metrics in zip(matching_nodes, metrics):
                try:
                    reflection = {
                        "node_id": node.node_id,
                        "content": node.text_content,
                        "relevance": node_metrics["final_score"],
                        "timestamp": node.timestamp,
                        "connected_memories": []
                    }
                    
                    # Get connected nodes with proper error handling
                    connections = await self.cognitive_network.traverse_network(
                        start_node=node,
                        max_depth=depth
                    )
                    
                    # Ensure we have a proper dict of depth -> nodes
                    if isinstance(connections, dict):
                        for depth_level, nodes_list in connections.items():
                            if not isinstance(nodes_list, (list, tuple)):
                                self.logger.warning(f"Expected list for depth {depth_level}, got {type(nodes_list)}")
                                continue
                                
                            for connected_node in nodes_list:
                                try:
                                    # Safely get connection weight
                                    connection_weight = (
                                        node.connections.get(str(connected_node.node_id), 0.0)
                                        if hasattr(node, 'connections') and node.connections 
                                        else 0.0
                                    )
                                    
                                    # Calculate echo intensity
                                    echo_intensity = connection_weight * node_metrics["final_score"]
                                    
                                    # Trigger echo effect
                                    await self.echo_manager.trigger_echo(
                                        node=connected_node,
                                        intensity=echo_intensity,
                                        comparison_state=context,
                                        query_text=topic
                                    )
                                    
                                    # Add to reflection results
                                    reflection["connected_memories"].append({
                                        "node_id": str(connected_node.node_id),
                                        "content": getattr(connected_node, "text_content", "No content available"),
                                        "connection_strength": connection_weight,
                                        "depth": depth_level
                                    })
                                except Exception as e:
                                    self.logger.error(f"Error processing connected node: {e}")
                                    continue
                    
                    results.append(reflection)
                    
                except Exception as node_error:
                    self.logger.error(f"Error processing node {node.node_id}: {node_error}")
                    continue
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            return []

    async def meditate_on_memory(
        self,
        memory_id: str,
        intensity: float = 0.8,
        duration: int = 5
    ) -> Dict[str, Any]:
        """
        Deep focus on a specific memory with amplified echo effects.
        """
        try:
            node = await self.get_node_by_id(memory_id)
            if not node:
                raise ValueError(f"Memory {memory_id} not found")
            node.last_accessed = time.time()
            context = await self.internal_context.get_memory_context(is_cognitive=True)
            base_intensity = min(1.0, max(0.1, intensity))
            echo_intensity = base_intensity * 1.5
            echo_effects = await self.echo_manager.trigger_echo(
                node=node,
                intensity=echo_intensity,
                comparison_state=context,
                duration_multiplier=duration
            )
            connections = await self.cognitive_network.traverse_network(
                start_node=node,
                max_depth=2,
            )
            result = {
                "memory_id": node.node_id,
                "content": getattr(node, "text_content", None),
                "meditation_intensity": base_intensity,
                "duration": duration,
                "echo_effects": echo_effects,
                "connected_memories": []
            }
            for depth_level, nodes_list in connections.items():
                for connected in nodes_list:
                    connection_weight = node.connections.get(connected.node_id, 0)
                    await self.echo_manager.trigger_echo(
                        node=connected,
                        intensity=base_intensity * connection_weight,
                        comparison_state=context,
                        duration_multiplier=max(1, duration // 2)
                    )
                    result["connected_memories"].append({
                        "node_id": connected.node_id,
                        "content": getattr(connected, "text_content", None),
                        "connection_strength": connection_weight,
                        "depth": depth_level
                    })
            return result
        except Exception as e:
            logger.error(f"Meditation failed: {e}")
            return {"error": str(e)}

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
            logger.info(f"Successfully processed conflict resolution between nodes {node_a_id} and {node_b_id}")
        except Exception as e:
            logger.error(f"Failed to handle conflict resolution: {e}")

    # -----------------------------
    # Performance Monitoring
    # -----------------------------
    async def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics for monitoring
        """
        stats = {
            'throttling_status': {},
            'cache_performance': {},
            'system_health': {}
        }
        
        try:
            # Throttling effectiveness
            current_time = time.time()
            recent_updates = 0
            stale_nodes = 0
            
            # Check cognitive nodes
            for node in self.cognitive_network.nodes.values():
                if node.last_connection_update:
                    if (current_time - node.last_connection_update) < 3600:  # Updated in last hour
                        recent_updates += 1
                    else:
                        stale_nodes += 1
            
            stats['throttling_status'] = {
                'nodes_updated_recently': recent_updates,
                'nodes_with_stale_connections': stale_nodes,
                'throttling_working': stale_nodes > recent_updates * 2  # More stale than recent = good
            }
            
            # Cache performance
            if hasattr(self.metrics_orchestrator, 'get_cache_stats'):
                stats['cache_performance'] = self.metrics_orchestrator.get_cache_stats()
            
            # System health
            stats['system_health'] = {
                'total_cognitive_nodes': len(self.cognitive_network.nodes),
                'total_body_nodes': len(self.body_network.nodes),
                'active_cognitive_nodes': len([n for n in self.cognitive_network.nodes.values() if not n.ghosted]),
                'active_body_nodes': len([n for n in self.body_network.nodes.values() if not n.ghosted])
            }
            
        except Exception as e:
            stats['error'] = str(e)
        
        return stats
    
    async def debug_cache_performance(self) -> None:
        """Debug method to monitor cache effectiveness."""
        try:
            if hasattr(self.metrics_orchestrator.calculate_metrics, 'cache'):
                cache_stats = self.metrics_orchestrator.calculate_metrics.cache.get_stats()
                self.logger.info(f"Metrics cache stats: {cache_stats}")
                
                # Trigger cache cleanup if hit rate is low
                if cache_stats.get('hit_rate', 0) < 0.3:
                    self.logger.warning("Low cache hit rate detected - clearing cache")
                    await self.metrics_orchestrator.clear_metrics_cache()
            
        except Exception as e:
            self.logger.error(f"Cache debugging failed: {e}")

    def get_connection_health_stats(self) -> Dict[str, Any]:
        """Get statistics about connection health across the network."""
        stats = {
            'cognitive_network': {},
            'body_network': {}
        }
        
        try:
            # Cognitive network stats
            if hasattr(self, 'cognitive_network') and self.cognitive_network:
                cog_nodes = self.cognitive_network.nodes.values()
                connection_counts = [len(node.connections) for node in cog_nodes if not node.ghosted]
                if connection_counts:
                    stats['cognitive_network'] = {
                        'total_nodes': len(connection_counts),
                        'avg_connections': sum(connection_counts) / len(connection_counts),
                        'max_connections': max(connection_counts),
                        'min_connections': min(connection_counts),
                        'nodes_at_limit': sum(1 for count in connection_counts if count >= 50),
                        'connection_distribution': {
                            '0-10': sum(1 for c in connection_counts if c <= 10),
                            '11-25': sum(1 for c in connection_counts if 11 <= c <= 25),
                            '26-40': sum(1 for c in connection_counts if 26 <= c <= 40),
                            '41-50': sum(1 for c in connection_counts if 41 <= c <= 50),
                            '50+': sum(1 for c in connection_counts if c > 50)
                        }
                    }
            
            # Body network stats  
            if hasattr(self, 'body_network') and self.body_network:
                body_nodes = self.body_network.nodes.values()
                connection_counts = [len(node.connections) for node in body_nodes if not node.ghosted]
                if connection_counts:
                    stats['body_network'] = {
                        'total_nodes': len(connection_counts),
                        'avg_connections': sum(connection_counts) / len(connection_counts),
                        'max_connections': max(connection_counts),
                        'min_connections': min(connection_counts),
                        'nodes_at_limit': sum(1 for count in connection_counts if count >= 50),
                        'connection_distribution': {
                            '0-10': sum(1 for c in connection_counts if c <= 10),
                            '11-25': sum(1 for c in connection_counts if 11 <= c <= 25),
                            '26-40': sum(1 for c in connection_counts if 26 <= c <= 40),
                            '41-50': sum(1 for c in connection_counts if 41 <= c <= 50),
                            '50+': sum(1 for c in connection_counts if c > 50)
                        }
                    }
                    
        except Exception as e:
            stats['error'] = str(e)
        
        return stats

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
            
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                logger.info("Update task was cancelled during shutdown.")

        try:
            await self.cognitive_network.shutdown()
            await self.body_network.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down networks: {e}")
                
        logger.info("Memory system shutdown complete.")

