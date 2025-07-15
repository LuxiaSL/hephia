"""
brain/cognition/memory/significance.py

Significance analysis for memory formation across different
interaction types. Supports both heuristic and metrics-based evaluation
using event-driven architecture for clean separation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

from brain.commands.model import ParsedCommand
from config import Config
from event_dispatcher import global_event_dispatcher, Event
from loggers import BrainLogger

class SourceType(Enum):
    COMMAND = "command"        # ExoProcessor commands
    DIRECT_CHAT = "direct"     # User interface chat
    DISCORD = "discord"        # Discord messages
    ENVIRONMENT = "environment"  # Environment transitions

@dataclass
class MemoryData:
    """Standardized structure for memory check events."""
    interface_id: str
    content: str               # The main content to be remembered
    context: Dict[str, Any]    # Current cognitive state/context
    source_type: SourceType
    metadata: Dict[str, Any]   # Source-specific metadata
    timestamp: datetime = field(default_factory=datetime.now)

    def to_event_data(self) -> Dict[str, Any]:
        """Convert to an event-safe dictionary structure."""
        return {
            "interface_id": self.interface_id,
            "content": self.content,
            "context": self.context,
            "source_type": self.source_type.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_event_data(cls, data: Dict[str, Any]) -> 'MemoryData':
        """Reconstruct a MemoryData object from event data."""
        return cls(
            interface_id=data["interface_id"],
            content=data["content"],
            context=data["context"],
            source_type=SourceType(data["source_type"]),
            metadata=data["metadata"],
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass
class SignificanceEvaluationRequest:
    """Request for metrics-based significance evaluation."""
    memory_data: MemoryData
    generated_content: str
    request_id: str
    timeout: float = 10.0

@dataclass 
class SignificanceEvaluationResponse:
    """Response from metrics-based significance evaluation."""
    request_id: str
    significance_score: float
    component_scores: Dict[str, float]
    evaluation_method: str  # "metrics" or "heuristic_fallback"
    error: Optional[str] = None

class SignificanceAnalyzer:
    """
    Significance analyzer supporting both heuristic and metrics-based evaluation.
    Uses event-driven architecture to access memory metrics without direct dependencies.
    """
    
    def __init__(self):
        """Initialize with configuration for different evaluation methods."""
        # Heuristic thresholds
        self.heuristic_thresholds: Dict[str, float] = {
            "exo_processor": Config.MEMORY_SIGNIFICANCE_THRESHOLD,
            "discord": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75,
            "user": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75
        }
        
        # Metrics-based configuration per source type
        self.metrics_config: Dict[str, Dict[str, float]] = {
            "discord": {
                "heuristic_weight": 0.15,
                "metrics_weight": 0.85,
                "metrics_threshold": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75,
            },
            "exo_processor": {
                "heuristic_weight": 0.15, 
                "metrics_weight": 0.85,
                "metrics_threshold": Config.MEMORY_SIGNIFICANCE_THRESHOLD
            },
            "user": {
                "heuristic_weight": 0.15,
                "metrics_weight": 0.85,
                "metrics_threshold": Config.MEMORY_SIGNIFICANCE_THRESHOLD * 0.75
            },
            "environment": {
                "heuristic_weight": 0.3,
                "metrics_weight": 0.7,
                "metrics_threshold": 0.6
            }
        }
        
        # Event response tracking
        self._pending_evaluations: Dict[str, asyncio.Future] = {}
        self._setup_event_listeners()
        
        BrainLogger.info("Enhanced SignificanceAnalyzer initialized with metrics support")

    def _setup_event_listeners(self) -> None:
        """Setup listeners for metrics evaluation responses."""
        global_event_dispatcher.add_listener(
            "significance:metrics_evaluation_response",
            self._handle_metrics_response
        )

    async def analyze_significance(
        self, 
        memory_data: MemoryData,
        generated_content: Optional[str] = None
    ) -> bool:
        """
        Analyze significance using combined heuristic and metrics approach.
        
        Args:
            memory_data: Memory data to evaluate
            generated_content: Optional LLM-generated content for metrics evaluation
            
        Returns:
            bool: Whether the memory is significant enough to store
        """
        try:
            # Get configuration for this source type
            source_config = self.metrics_config.get(
                memory_data.interface_id, 
                self.metrics_config.get("user", {})  # Default fallback
            )
            
            # Always calculate heuristic score as baseline
            heuristic_score = self._calculate_heuristic_significance(memory_data)
            
            # If no generated content provided, fall back to heuristic only
            if generated_content is None:
                BrainLogger.debug(f"No generated content - using heuristic only for {memory_data.interface_id}")
                threshold = self.heuristic_thresholds.get(memory_data.interface_id, 0.5)
                return heuristic_score > threshold
            
            # Attempt metrics-based evaluation if configured
            metrics_weight = source_config.get("metrics_weight", 0.0)
            if metrics_weight > 0:
                try:
                    metrics_score = await self._evaluate_via_metrics(
                        memory_data, 
                        generated_content,
                        timeout=source_config.get("timeout", 10.0)
                    )
                    
                    if metrics_score is not None:
                        # Combine heuristic and metrics scores
                        heuristic_weight = source_config.get("heuristic_weight", 1.0)
                        combined_score = (
                            heuristic_score * heuristic_weight + 
                            metrics_score * metrics_weight
                        )
                        
                        # Normalize by total weight
                        total_weight = heuristic_weight + metrics_weight
                        final_score = combined_score / total_weight if total_weight > 0 else heuristic_score
                        
                        threshold = source_config.get("metrics_threshold", 0.6)
                        
                        BrainLogger.debug(
                            f"Combined significance for {memory_data.interface_id}: "
                            f"heuristic={heuristic_score:.3f}, metrics={metrics_score:.3f}, "
                            f"final={final_score:.3f}, threshold={threshold:.3f}"
                        )
                        
                        return final_score > threshold
                        
                except Exception as e:
                    BrainLogger.warning(f"Metrics evaluation failed for {memory_data.interface_id}: {e}")
                    # Fall through to heuristic fallback
            
            # Fallback to heuristic evaluation
            BrainLogger.debug(f"Using heuristic fallback for {memory_data.interface_id}")
            threshold = self.heuristic_thresholds.get(memory_data.interface_id, 0.5)
            return heuristic_score > threshold
            
        except Exception as e:
            BrainLogger.error(f"Significance analysis failed for {memory_data.interface_id}: {e}")
            # Emergency fallback to permissive threshold
            return True

    async def _evaluate_via_metrics(
        self,
        memory_data: MemoryData,
        generated_content: str,
        timeout: float = 10.0
    ) -> Optional[float]:
        """
        Request metrics-based evaluation via event system.
        
        Args:
            memory_data: Memory data to evaluate
            generated_content: LLM-generated content
            timeout: Maximum time to wait for response
            
        Returns:
            Optional[float]: Metrics-based significance score, None if failed
        """
        try:
            # Generate unique request ID
            request_id = f"sig_eval_{memory_data.interface_id}_{datetime.now().timestamp()}"
            
            # Create future for response tracking
            response_future = asyncio.Future()
            self._pending_evaluations[request_id] = response_future
            
            # Dispatch evaluation request event
            evaluation_request = SignificanceEvaluationRequest(
                memory_data=memory_data,
                generated_content=generated_content,
                request_id=request_id,
                timeout=timeout
            )
            
            global_event_dispatcher.dispatch_event(Event(
                "significance:request_metrics_evaluation",
                {
                    "request_id": request_id,
                    "memory_data": memory_data.to_event_data(),
                    "generated_content": generated_content,
                    "timeout": timeout
                }
            ))
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                
                if response.error:
                    BrainLogger.warning(f"Metrics evaluation error: {response.error}")
                    return None
                    
                BrainLogger.debug(
                    f"Metrics evaluation completed: score={response.significance_score:.3f}, "
                    f"method={response.evaluation_method}"
                )
                return response.significance_score
                
            except asyncio.TimeoutError:
                BrainLogger.warning(f"Metrics evaluation timeout for request {request_id}")
                return None
                
        except Exception as e:
            BrainLogger.error(f"Failed to request metrics evaluation: {e}")
            return None
            
        finally:
            # Cleanup
            self._pending_evaluations.pop(request_id, None)

    def _handle_metrics_response(self, event: Event) -> None:
        """Handle metrics evaluation response events."""
        try:
            request_id = event.data.get("request_id")
            if not request_id or request_id not in self._pending_evaluations:
                return
                
            future = self._pending_evaluations[request_id]
            if future.done():
                return
                
            # Create response object
            response = SignificanceEvaluationResponse(
                request_id=request_id,
                significance_score=event.data.get("significance_score", 0.0),
                component_scores=event.data.get("component_scores", {}),
                evaluation_method=event.data.get("evaluation_method", "unknown"),
                error=event.data.get("error")
            )
            
            # Complete the future
            future.set_result(response)
            
        except Exception as e:
            BrainLogger.error(f"Failed to handle metrics response: {e}")
            # Try to fail the future if possible
            request_id = event.data.get("request_id")
            if request_id in self._pending_evaluations:
                future = self._pending_evaluations[request_id]
                if not future.done():
                    future.set_exception(e)

    def _calculate_heuristic_significance(self, memory_data: MemoryData) -> float:
        """
        Calculate significance using simple heuristic methods.
        """
        if memory_data.source_type == SourceType.COMMAND:
            return self._analyze_command_significance(memory_data)
        elif memory_data.source_type == SourceType.DISCORD:
            return self._analyze_social_significance(memory_data)
        elif memory_data.source_type == SourceType.DIRECT_CHAT:
            return self._analyze_social_significance(memory_data)
        elif memory_data.source_type == SourceType.ENVIRONMENT:
            return self._analyze_environment_significance(memory_data)
        else:
            BrainLogger.warning(f"Unknown source type: {memory_data.source_type}")
            return 0.5  # Neutral score for unknown types

    def _analyze_command_significance(self, data: MemoryData) -> float:
        score = 0.0
        metadata = data.metadata

        # Command complexity (0.3)
        command = metadata.get('command')
        if isinstance(command, ParsedCommand) and command.parameters:
            # Each parameter contributes 0.1, up to 0.3 maximum.
            score += min(0.3, len(command.parameters) * 0.1)

        # Response impact (0.3)
        response = metadata.get('response', '')
        if isinstance(response, str):
            words = response.split()
            # Up to 0.15 for word count impact.
            score += min(0.15, len(words) / 100)
        # Additional impact if the command did not succeed.
        if not metadata.get('success', True):
            score += 0.15

        # Environment context (0.2)
        if command and getattr(command, 'environment', None):
            score += 0.2

        # Result impact (0.2)
        result = metadata.get('result')
        if result:
            # Coerce potential None values to defaults.
            message = (getattr(result, 'message', '') or '')
            data_val = (getattr(result, 'data', {}) or {})
            state_changes = (getattr(result, 'state_changes', {}) or {})
            
            impact_score = (len(message.split()) + len(data_val) * 2 + len(state_changes) * 3) / 25
            # Only add if the command wasn't a trivial help or version request.
            if command and command.action not in ['help', 'version', 'list']:
                score += min(0.2, impact_score)

        return score

    def _analyze_social_significance(self, data: MemoryData) -> float:
        """
        Analyze significance of social interactions with path-based channel references.
        """
        score = 0.0
        metadata = data.metadata
        
        if data.source_type == SourceType.DISCORD:
            # Message length (0.5)
            message = metadata.get('message', {}).get('content', '')
            msg_words = message.split()
            score += min(0.5, len(msg_words) / 50)

            # Interaction depth (0.5)
            if len(metadata.get('history', [])) >= 2:
                score += 0.25
            if metadata.get('mentions_bot'):
                score += 0.25
        else:  # DIRECT_CHAT
            # Conversation depth (0.5)
            conv_data = metadata.get('conversation', {})
            if conv_data.get('has_multi_turn'):
                score += 0.2
            score += min(0.2, conv_data.get('total_messages', 0) * 0.05)

            # Message content (0.5)
            if last_msg := conv_data.get('last_user_message'):
                score += min(0.5, len(last_msg.split()) / 50)

        score = min(score, 1.0)
        return score

    def _analyze_environment_significance(self, data: MemoryData) -> float:
        score = 0.0
        metadata = data.metadata
        
        # Session length (0.4)
        history = metadata.get('history', [])
        score += min(0.4, len(history) * 0.1)
        
        # Command variety (0.3)
        unique_commands = len(set(
            cmd.get('action') for cmd in history 
            if isinstance(cmd, dict) and 'action' in cmd
        ))
        score += min(0.3, unique_commands * 0.1)
        
        # Success rate (0.3)
        successes = sum(1 for cmd in history if cmd.get('success', False))
        if history:
            score += 0.3 * (successes / len(history))
            
        return score

    def update_configuration(
        self, 
        source_type: str, 
        config: Dict[str, Union[float, Dict[str, float]]]
    ) -> None:
        """
        Update configuration for a specific source type.
        
        Args:
            source_type: Interface/source identifier
            config: Configuration updates
        """
        try:
            if source_type not in self.metrics_config:
                self.metrics_config[source_type] = {}
                
            self.metrics_config[source_type].update(config)
            BrainLogger.info(f"Updated significance config for {source_type}: {config}")
            
        except Exception as e:
            BrainLogger.error(f"Failed to update significance config: {e}")

    def get_configuration(self, source_type: str) -> Dict[str, Any]:
        """Get current configuration for a source type."""
        return {
            "heuristic_threshold": self.heuristic_thresholds.get(source_type, 0.5),
            "metrics_config": self.metrics_config.get(source_type, {})
        }