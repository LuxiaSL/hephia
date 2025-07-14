"""
brain/cognition/memory/manager.py

Central memory management system with neuromorphic flow.
Generates memory content FIRST, then evaluates significance using
sophisticated metrics on the actual generated content.

Event → Generate Content → Sophisticated Significance Check → Form or Discard Memory
"""

from typing import Dict, Any, Optional, List
import asyncio

from brain.utils.tracer import brain_trace
from brain.interfaces.base import CognitiveInterface
from brain.cognition.memory.significance import SignificanceAnalyzer, MemoryData
from brain.prompting.loader import get_prompt
from core.state_bridge import StateBridge
from internal.modules.cognition.cognitive_bridge import CognitiveBridge
from api_clients import APIManager
from loggers import BrainLogger
from config import Config
from event_dispatcher import global_event_dispatcher, Event

class MemoryManager:
    def __init__(
        self,
        state_bridge: StateBridge,
        cognitive_bridge: CognitiveBridge,
        api: APIManager,
        interfaces: Dict[str, CognitiveInterface]
    ):
        self.state_bridge = state_bridge
        self.cognitive_bridge = cognitive_bridge
        self.api = api
        self.interfaces = interfaces
        self.significance_analyzer = SignificanceAnalyzer()
        self._memory_lock = asyncio.Lock()
        
        self._generation_stats = {
            "total_generated": 0,
            "accepted": 0,
            "rejected": 0,
            "generation_failures": 0,
            "significance_failures": 0
        }
        
        self._setup_memory_listeners()

    def _setup_memory_listeners(self):
        """Setup listeners for memory checks from all interfaces."""
        for interface_id in self.interfaces:
            global_event_dispatcher.add_listener(
                f"cognitive:{interface_id}:memory_check",
                self._handle_memory_event
            )
    
    @brain_trace
    async def _handle_memory_event(self, event: Event) -> None:
        """
        Handle memory check events with neuromorphic flow.
        
        1. Extract memory data
        2. Generate memory content (always - the "experience")
        3. Evaluate significance using sophisticated metrics on generated content
        4. Form memory if significant, discard if not
        """
        try:
            # Extract memory data from event
            if isinstance(event.data, dict) and "event_data" in event.data:
                memory_payload = event.data["event_data"]
            else:
                memory_payload = event.data
            
            memory_data = MemoryData.from_event_data(memory_payload)

            brain_trace.memory.event(
                event_type=event.data.get("event_type"),
                interface_id=memory_data.interface_id
            )
            
            brain_trace.memory.generate("Generating memory content")
            BrainLogger.debug(f"Experiencing memory content for {memory_data.interface_id}")
            
            memory_content = await self._generate_memory_content(memory_data)
            self._generation_stats["total_generated"] += 1
            
            if not memory_content:
                brain_trace.memory.skip(reason="content generation failed")
                self._generation_stats["generation_failures"] += 1
                BrainLogger.warning(f"Memory content generation failed for {memory_data.interface_id}")
                return
            
            brain_trace.memory.significance("Evaluating significance with generated content")
            is_significant = await self._check_significance_with_content(memory_data, memory_content)
            
            if not is_significant:
                brain_trace.memory.skip(reason="below significance threshold after content evaluation")
                self._generation_stats["rejected"] += 1
                BrainLogger.debug(
                    f"Generated memory content rejected for {memory_data.interface_id} "
                    f"(length: {len(memory_content)} chars)"
                )
                # Content is discarded here (neuromorphic pruning)
                return
            
            # Content passed significance check - form the memory
            brain_trace.memory.form("Forming memory after significance approval")
            self._generation_stats["accepted"] += 1
            BrainLogger.info(
                f"Memory content accepted for {memory_data.interface_id} "
                f"(length: {len(memory_content)} chars)"
            )
            
            await self._form_memory(memory_data, memory_content)
                
        except Exception as e:
            self._generation_stats["significance_failures"] += 1
            brain_trace.error(
                error=e,
                context={
                    "event_data": event.data,
                    "interface_id": memory_data.interface_id if 'memory_data' in locals() else None,
                    "has_content": 'memory_content' in locals(),
                    "content_length": len(memory_content) if 'memory_content' in locals() and memory_content else 0
                }
            )
            BrainLogger.error(f"Memory event handling failed: {e}")
            raise

    async def _check_significance_with_content(
        self, 
        memory_data: MemoryData, 
        generated_content: str
    ) -> bool:
        """
        Enhanced significance checking that includes generated content.
        Uses the new SignificanceAnalyzer with metrics-based evaluation.
        
        Args:
            memory_data: Original memory context and metadata
            generated_content: LLM-generated memory content to evaluate
            
        Returns:
            bool: Whether the memory is significant enough to store
        """
        try:
            # Call enhanced SignificanceAnalyzer with generated content
            return await self.significance_analyzer.analyze_significance(
                memory_data=memory_data,
                generated_content=generated_content
            )
        except Exception as e:
            BrainLogger.error(f"Error in enhanced significance checking: {e}")
            # Fallback to heuristic-only evaluation on error
            try:
                BrainLogger.warning(f"Falling back to heuristic-only significance for {memory_data.interface_id}")
                return await self.significance_analyzer.analyze_significance(
                    memory_data=memory_data,
                    generated_content=None  # Forces heuristic fallback
                )
            except Exception as fallback_error:
                BrainLogger.error(f"Even heuristic fallback failed: {fallback_error}")
                # Emergency permissive fallback
                return True
    
    @brain_trace
    async def _generate_memory_content(self, memory_data: MemoryData) -> Optional[str]:
        """
        Generate memory content based on source type.
        This happens BEFORE significance evaluation.
        """
        try:
            interface = self.interfaces.get(memory_data.interface_id)
            if not interface:
                brain_trace.memory.error(error="Interface not found", interface_id=memory_data.interface_id)
                return None

            # Get formatted memory prompt from interface
            brain_trace.memory.format("Getting memory prompt")
            BrainLogger.debug(f"Getting memory prompt for {memory_data.interface_id}")
            memory_prompt = await interface.format_memory_context(
                content=memory_data.content,
                state=memory_data.context,
                metadata=memory_data.metadata
            )

            # Generate memory content using LLM
            brain_trace.memory.model("Getting model config")
            model_name = Config.get_cognitive_model()
            model_config = Config.AVAILABLE_MODELS[model_name]
            
            brain_trace.memory.completion("Generating completion")
            BrainLogger.debug(f"Getting LLM completion for memory from {memory_data.interface_id}")
            content = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {
                        "role": "system",
                        "content": get_prompt(
                            "memory.formation.system",
                            model=model_name
                        )
                    },
                    {
                        "role": "user",
                        "content": memory_prompt
                    }
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )

            if content:
                brain_trace.memory.success(content_length=len(content))
                BrainLogger.debug(f"Generated {len(content)} chars for {memory_data.interface_id}")
            else:
                brain_trace.memory.fail(reason="empty content")
                # Try fallback memory from interface
                content = await interface.get_fallback_memory(memory_data)
                if content:
                    BrainLogger.info(f"Using fallback memory content for {memory_data.interface_id}")
                
            return content
            
        except Exception as e:
            brain_trace.error(
                error=e,
                context={
                    "interface_id": memory_data.interface_id,
                    "content_length": len(memory_data.content) if memory_data.content else 0,
                    "metadata": memory_data.metadata
                }
            )
            BrainLogger.error(f"Memory content generation failed for {memory_data.interface_id}: {e}")
            return None
    
    @brain_trace
    async def _form_memory(self, memory_data: MemoryData, content: str) -> None:
        """
        Form final memory and dispatch to cognitive bridge.
        This now only happens for content that passed sophisticated significance evaluation.
        """
        try:
            event_data = {
                "event_type": memory_data.interface_id,
                "event_data": {
                    "content": content,
                    "context": memory_data.context,
                    "metadata": {
                        "source_type": memory_data.source_type.value,
                        "original_metadata": memory_data.metadata,
                        "timestamp": memory_data.timestamp.isoformat(),
                        "neuromorphic_flow": True,  # Flag indicating this used new flow
                        "content_length": len(content)
                    }
                }
            }

            brain_trace.memory.dispatch(
                event_type="memory_formation",
                content_length=len(content),
                interface=memory_data.interface_id
            )
            
            global_event_dispatcher.dispatch_event(Event(
                "cognitive:memory:request_formation",
                event_data
            ))
        except Exception as e:
            brain_trace.error(
                error=e,
                context={
                    "content_length": len(content),
                    "interface_id": memory_data.interface_id,
                    "source_type": memory_data.source_type.value
                }
            )
            raise
    
    @brain_trace
    async def handle_conflict(
        self,
        node_a_id: str,
        node_b_id: str,
        conflicts: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> None:
        """
        Handle conflicts between memories.
        Uses LLM to synthesize conflicting memories.
        """
        try:
            # Get node data
            brain_trace.conflict.retrieve("Getting node data")
            nodeA = await self.cognitive_bridge._get_node_by_id(node_a_id)
            nodeB = await self.cognitive_bridge._get_node_by_id(node_b_id)

            if not nodeA or not nodeB:
                brain_trace.conflict.error(
                    error="Missing nodes",
                    context={"node_a": bool(nodeA), "node_b": bool(nodeB)}
                )
                return

            # Format conflict resolution prompt
            brain_trace.conflict.prompt("Formatting resolution prompt")
            prompt = self._format_conflict_prompt(
                nodeA.content,
                nodeB.content,
                conflicts,
                metrics
            )

            # Get resolution
            brain_trace.conflict.resolve("Getting resolution")
            synthesis = await self._get_conflict_content(prompt)
            
            if synthesis:
                brain_trace.conflict.dispatch(
                    resolution="synthesis",
                    content_length=len(synthesis)
                )
                
                global_event_dispatcher.dispatch_event(Event(
                    "cognitive:memory:conflict_resolved",
                    {
                        "node_a_id": node_a_id,
                        "node_b_id": node_b_id,
                        "synthesis_text": synthesis,
                        "resolution_context": {
                            "conflicts": conflicts,
                            "metrics": metrics,
                            "resolution_type": "synthesis"
                        }
                    }
                ))

        except Exception as e:
            brain_trace.error(
                error=e,
                context={
                    "node_a_id": node_a_id,
                    "node_b_id": node_b_id,
                    "conflict_types": list(conflicts.keys()),
                    "metric_types": list(metrics.keys())
                }
            )
            raise

    def _format_conflict_prompt(
        self,
        content_a: str,
        content_b: str,
        conflicts: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> str:
        """Format prompt for memory conflict resolution."""
        conflict_details = []
        max_state_conflict_severity = 0.0
        has_state_conflict = False

        if not conflicts:
            conflict_details.append("No specific conflict details identified, but divergence threshold met.")
        else:
            for divergence in conflicts:
                subtype = divergence.get('subtype', 'Unknown divergence')
                severity = divergence.get('severity', 0.0)
                context = divergence.get('context', 'No specific context provided.')

                # Format the string for the prompt
                details_str = f"Type: {subtype}, Severity: {severity:.2f}"
                if context != 'No specific context provided.':
                     details_str += f", Context: {context}"
                conflict_details.append(details_str)

                # Track state conflicts for the state_metrics placeholder
                if subtype.startswith('state_'):
                    has_state_conflict = True
                    max_state_conflict_severity = max(max_state_conflict_severity, severity)
        conflict_details_text = "\\n".join(f"- {detail}" for detail in conflict_details)

        # --- Extract metrics for placeholders ---
        component_metrics = metrics.get('component_metrics', {})

        # Semantic Metrics: Use embedding similarity
        semantic_metrics_val = component_metrics.get('semantic', {}).get('embedding_similarity', None)
        semantic_metrics_str = f"{semantic_metrics_val:.2f}" if isinstance(semantic_metrics_val, float) else "N/A"

        # Emotional Metrics: Use vector similarity (lower means less aligned)
        emotional_metrics_val = component_metrics.get('emotional', {}).get('vector_similarity', None)
        emotional_metrics_str = f"{emotional_metrics_val:.2f}" if isinstance(emotional_metrics_val, float) else "N/A"

        # State Metrics: Use info derived from state conflicts
        if has_state_conflict:
            # Represent consistency as inverse of max conflict severity
            state_consistency_est = 1.0 - max_state_conflict_severity
            state_metrics_str = f"~{state_consistency_est:.2f} (estimated based on max conflict)"
        elif component_metrics.get('state'):
             # If state metrics exist but no specific conflict listed, assume high consistency
             state_metrics_str = "~0.90+ (no major state conflicts detected)"
        else:
             state_metrics_str = "N/A (State metrics unavailable)"

        # --- Get prompt using extracted variables ---
        try:
            prompt_text = get_prompt(
                "memory.conflict.user",
                model=Config.get_cognitive_model(), # Or appropriate model for this task
                vars={
                    "content_a": content_a,
                    "content_b": content_b,
                    "conflict_details_text": conflict_details_text,
                    "semantic_metrics": semantic_metrics_str,
                    "emotional_metrics": emotional_metrics_str,
                    "state_metrics": state_metrics_str
                }
            )
            return prompt_text
        except Exception as e:
            BrainLogger.error(f"Failed to format conflict prompt using template: {e}")
            return f"Memory A: {content_a}\\nMemory B: {content_b}\\nConflicts:\\n{conflict_details_text}\\nMetrics: Sem={semantic_metrics_str}, Emo={emotional_metrics_str}, State={state_metrics_str}\\nPlease synthesize."

    async def _get_conflict_content(self, prompt: str) -> Optional[str]:
        """
        Get one-turn LLM completion for memory formation.
        """
        try:
            model_name = Config.get_summary_model()
            model_config = Config.AVAILABLE_MODELS[model_name]

            result = await self.api.create_completion(
                provider=model_config.provider.value,
                model=model_config.model_id,
                messages=[
                    {"role": "system", "content": get_prompt(
                        "memory.conflict.system",
                        model=model_name
                    )},
                    {"role": "user", "content": prompt}
                ],
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                return_content_only=True
            )

            return result if result and len(result.strip()) > 20 else None

        except Exception as e:
            BrainLogger.error(f"Memory content generation failed: {e}")
            return None

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the neuromorphic memory generation flow."""
        total = self._generation_stats["total_generated"]
        if total == 0:
            return {**self._generation_stats, "acceptance_rate": 0.0, "rejection_rate": 0.0}
        
        return {
            **self._generation_stats,
            "acceptance_rate": self._generation_stats["accepted"] / total,
            "rejection_rate": self._generation_stats["rejected"] / total
        }

    def reset_generation_stats(self) -> None:
        """Reset generation statistics (useful for monitoring)."""
        self._generation_stats = {
            "total_generated": 0,
            "accepted": 0,
            "rejected": 0,
            "generation_failures": 0,
            "significance_failures": 0
        }
        BrainLogger.info("Memory generation statistics reset")