"""
environments/meditation.py - Meditation environment for Hephia.

Enables focused introspection and state influence through meditation
commands. Used to elicit specific emotional/cognitive states and deepen
internal awareness.

todo in both: error handling & failure return checks
"""

from typing import Dict, List, Any
from brain.commands.model import (
    CommandDefinition,
    Parameter,
    Flag,
    ParameterType,
    CommandResult
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
        Use 'focus' to meditate on specific states or concepts.
        Use 'absorb' to deeply reflect on specific memories.
        
        Examples:
        - meditate focus "calm"
        - meditate focus "contentment"
        - meditate absorb <memory_id>
        """

    def _register_commands(self) -> None:
        """Register meditation environment commands."""
        self.register_command(
            CommandDefinition(
                name="focus",
                description="Meditate on and elicit a specific state",
                parameters=[
                    Parameter(
                        name="state",
                        description="State or concept to focus on",
                        required=True,
                        examples=[
                            '"calm"',
                            '"joy"',
                            '"clarity"'
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
                    'meditate focus "joy" --intensity=0.7',
                    'meditate focus "peace" --duration=2'
                ],
                related_commands=[
                    'meditate absorb',
                    'reflect query "similar feeling"'
                ],
                category="State Focus"
            )
        )

        self.register_command(
            CommandDefinition(
                name="absorb",
                description="Deeply absorb and reflect on a specific memory",
                parameters=[
                    Parameter(
                        name="memory_id",
                        description="ID of the memory to absorb",
                        required=True
                    )
                ],
                flags=[
                    Flag(
                        name="intensity",
                        description="Depth of absorption",
                        type=ParameterType.NUMBER,
                        default=0.5
                    )
                ],
                examples=[
                    'meditate absorb memory_123',
                    'meditate absorb memory_456 --intensity=0.8'
                ],
                related_commands=[
                    'reflect query "related memories"',
                    'meditate focus "similar feeling"'
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
                
                # Request state influence through cognitive bridge
                result = await self.cognitive_bridge.meditate_on_state(
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
                            'reflect query "similar states"'
                        ]
                    )
                
                response = (
                    f"Focusing on '{state}'...\n\n"
                    f"Feeling a {result.get('intensity', 'subtle')} sense of {state}.\n"
                    f"Internal state is shifting {result.get('direction', 'gradually')}."
                )
                
                # Suggest follow-ups based on the meditation result
                suggested = [
                    f'reflect query "times I felt {state}"',
                    f'meditate focus "{state}" --duration={duration + 1}'
                ]
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "state": state,
                        "intensity": intensity,
                        "duration": duration,
                        "result": result
                    }
                )
                
            elif action == "absorb":
                memory_id = params[0]
                intensity = flags.get("intensity", 0.5)
                
                # Request memory absorption through cognitive bridge
                result = await self.cognitive_bridge.absorb_memory(
                    memory_id,
                    intensity
                )
                
                if not result:
                    return CommandResult(
                        success=False,
                        message=f"Could not find or access memory {memory_id}.",
                        suggested_commands=[
                            'reflect recent',
                            'reflect query "recent memories"'
                        ]
                    )
                
                # Format the absorption experience
                memory_content = result.get("content", "this memory")
                effects = result.get("effects", [])
                
                response = (
                    f"Absorbing memory: {memory_content}\n\n"
                    + "\n".join(f"• {effect}" for effect in effects)
                )
                
                # Suggest relevant follow-ups
                suggested = [
                    f'reflect query "similar to {memory_id}"',
                    f'meditate focus "{result.get("dominant_feeling", "this feeling")}"'
                ]
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "memory_id": memory_id,
                        "intensity": intensity,
                        "result": result
                    }
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Meditation failed: {str(e)}",
                suggested_commands=[
                    'meditate focus "calm"',
                    'help'
                ],
                error=str(e)
            )