"""
environments/meditation.py - Meditation environment for Hephia.

Enables focused introspection and state influence through meditation
commands. Used to elicit specific emotional/cognitive states and deepen
internal awareness.
"""

from typing import Dict, List, Any
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult,
    CommandValidationError
)
from .base_environment import BaseEnvironment
from internal.modules.cognition.cognitive_bridge import CognitiveBridge

class MeditateEnvironment(BaseEnvironment):
    """
    Environment for meditation and focused internal state influence.
    Provides commands for eliciting specific states and deepening awareness.
    """
    
    def __init__(self, cognitive_bridge: CognitiveBridge):
        self.cognitive_bridge = cognitive_bridge
        super().__init__()
        
        self.help_text = """
The meditation environment allows focused introspection and state influence.
Use 'focus' to meditate on states using related memories for guidance.
Use 'absorb' to deeply reflect on memories about specific topics.

Examples:
- meditate focus "calm" - find memories of calm moments and focus toward that state
- meditate absorb "feeling accomplished" - absorb memories about accomplishment
- meditate focus "confidence" --intensity=0.7 --duration=2
- meditate absorb "peaceful moments" --count=2
"""

    def _register_commands(self) -> None:
        """Register meditation environment commands."""
        self.register_command(
            CommandDefinition(
                name="focus",
                description="Meditate on a state using related memories for emotional guidance",
                parameters=[
                    Parameter(
                        name="state",
                        description="State, feeling, or concept to focus on",
                        required=True,
                        examples=[
                            '"calm"',
                            '"confidence"',
                            '"clarity"',
                            '"feeling accomplished"'
                        ]
                    )
                ],
                flags=[
                    Flag(
                        name="intensity",
                        description="Desired intensity of the meditation",
                        type=ParameterType.NUMBER,
                        default=0.5
                    ),
                    Flag(
                        name="duration",
                        description="Meditation duration in cycles",
                        type=ParameterType.INTEGER,
                        default=1
                    )
                ],
                examples=[
                    'meditate focus "calm"',
                    'meditate focus "confidence" --intensity=0.7',
                    'meditate focus "feeling proud" --duration=2',
                    'meditate focus "inner peace" --intensity=0.6 --duration=3'
                ],
                related_commands=[
                    'meditate absorb "related topic"',
                    'reflect query "similar feeling"'
                ],
                category="State Focus"
            )
        )

        self.register_command(
            CommandDefinition(
                name="absorb",
                description="Deeply absorb and reflect on memories related to a topic",
                parameters=[
                    Parameter(
                        name="topic",
                        description="Topic or theme to search for and absorb memories about",
                        required=True,
                        examples=[
                            '"feeling accomplished"',
                            '"times I learned something new"',
                            '"peaceful moments"'
                        ]
                    )
                ],
                flags=[
                    Flag(
                        name="intensity",
                        description="Depth of absorption",
                        type=ParameterType.NUMBER,
                        default=0.8
                    ),
                    Flag(
                        name="count",
                        description="Maximum number of memories to absorb",
                        type=ParameterType.INTEGER,
                        default=3
                    )
                ],
                examples=[
                    'meditate absorb "feeling proud of my work"',
                    'meditate absorb "calm moments" --intensity=0.6',
                    'meditate absorb "recent insights" --count=2 --intensity=0.9'
                ],
                related_commands=[
                    'reflect query "similar topic"',
                    'meditate focus "related feeling"'
                ],
                category="Memory Absorption"
            )
        )
    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """Execute meditation commands."""
        try:
            if action == "focus":
                state = params[0]
                intensity = flags.get("intensity", 0.5)
                duration = flags.get("duration", 1)
                
                # Request memory-informed focus meditation through cognitive bridge
                result = await self.cognitive_bridge.focus_on_state_with_memories(
                    state,
                    intensity,
                    duration
                )
                
                if not result:
                    return CommandResult(
                        success=True,
                        message=f"Attempted to focus on '{state}' but felt no significant shift.",
                        suggested_commands=[
                            f'meditate focus "{state}" --intensity={min(1.0, intensity + 0.2)}',
                            f'reflect query "{state}"'
                        ]
                    )
                
                # Check if this was memory-informed or basic meditation
                memory_count = result.get('memory_count', 0)
                effects = result.get('effects', [])
                
                if memory_count > 0:
                    # Memory-informed meditation response
                    avg_direction = result.get('average_direction', {})
                    valence = avg_direction.get('valence', 0.0)
                    arousal = avg_direction.get('arousal', 0.0)
                    
                    # Create descriptive response based on actual emotional direction
                    direction_desc = ""
                    if abs(valence) > 0.1:
                        direction_desc += "uplifting" if valence > 0 else "contemplative"
                    if abs(arousal) > 0.1:
                        if direction_desc:
                            direction_desc += " and "
                        direction_desc += "energizing" if arousal > 0 else "calming"
                    if not direction_desc:
                        direction_desc = "subtle"
                    
                    response = (
                        f"Focusing on '{state}' through {memory_count} related memories...\n\n"
                        f"Drawing emotional patterns from past experiences of {state}.\n"
                        f"Feeling a {direction_desc} influence (valence: {valence:+.2f}, arousal: {arousal:+.2f}).\n\n"
                        f"Effects:\n" + "\n".join(f"• {effect}" for effect in effects)
                    )
                    
                    suggested = [
                        f'reflect query "times I felt {state}"',
                        f'meditate absorb "{state}"'
                    ]
                else:
                    # Basic meditation fallback response
                    response = (
                        f"Focusing on '{state}'...\n\n"
                        f"No specific memories surfaced, drawing from intuitive understanding.\n"
                        f"Gentle influence toward {state}."
                    )
                    
                    suggested = [
                        f'reflect query "{state}"',
                        f'meditate absorb "{state}"'
                    ]
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "state": state,
                        "intensity": intensity,
                        "duration": duration,
                        "memory_informed": memory_count > 0,
                        "result": result
                    }
                )
            # Handle memory absorption command        
            elif action == "absorb":
                topic = params[0]
                intensity = flags.get("intensity", 0.8)
                max_count = flags.get("count", 3)
                
                # Request memory absorption through cognitive bridge
                result = await self.cognitive_bridge.absorb_memory(
                    topic,  # Now passing topic instead of memory_id
                    intensity,
                    max_count
                )
                
                if not result:
                    error = CommandValidationError(
                        message=f"No memories found related to '{topic}'",
                        suggested_fixes=[
                            "Try a broader search term",
                            "Check recent memories first",
                            "Use different keywords"
                        ],
                        related_commands=[
                            'reflect query "' + topic + '"',
                            'reflect recent',
                            'meditate absorb "broader topic"'
                        ],
                        examples=[
                            'meditate absorb "feeling good"',
                            'meditate absorb "recent experiences"'
                        ]
                    )
                    return CommandResult(
                        success=False,
                        message=f"No memories found for '{topic}'. Try a broader search or check recent memories.",
                        suggested_commands=[
                            f'reflect query "{topic}"',
                            'reflect recent'
                        ],
                        error=error
                    )
                
                # Format the absorption experience with actual results
                absorbed_count = result.get("absorbed_count", 0)
                memories = result.get("memories", [])
                collective_effects = result.get("collective_effects", [])
                
                # Create detailed response showing what was absorbed
                memory_descriptions = []
                for memory in memories[:3]:  # Show top 3 in response
                    content_preview = memory['content'][:50] + "..." if len(memory['content']) > 50 else memory['content']
                    relevance = memory.get('relevance', 0.0)
                    memory_descriptions.append(f"• {content_preview} (relevance: {relevance:.2f})")
                
                effects_text = ""
                if collective_effects:
                    effects_text = "\n\nEffects:\n" + "\n".join(f"• {effect}" for effect in collective_effects[:5])
                
                response = (
                    f"Absorbed {absorbed_count} memories about '{topic}':\n\n" +
                    "\n".join(memory_descriptions) +
                    effects_text
                )
                
                # Suggest relevant follow-ups based on the topic
                suggested = [
                    f'reflect query "{topic}"',
                    f'meditate focus "{topic}"',
                    'reflect recent'
                ]
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "topic": topic,
                        "intensity": intensity,
                        "absorbed_count": absorbed_count,
                        "memories": memories,
                        "collective_effects": collective_effects
                    }
                )
                
        except Exception as e:
            error = CommandValidationError(
                message=str(e),
                suggested_fixes=[
                    "Try a lower intensity level",
                    "Start with basic meditation states",
                    "Check if the cognitive bridge is functioning"
                ],
                related_commands=[
                    'meditate focus "calm" --intensity=0.3',
                    'meditate focus "peace"',
                    'help'
                ],
                examples=[
                    'meditate focus "calm"',
                    'meditate focus "clarity" --intensity=0.5'
                ]
            )
            
            return CommandResult(
                success=False,
                message=f"Meditation failed: {str(e)}",
                suggested_commands=[
                    'meditate focus "calm"',
                    'help'
                ],
                error=error
            )