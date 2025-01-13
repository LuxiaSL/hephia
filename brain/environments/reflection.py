"""
reflection.py - Memory reflection environment for Hephia.

Enables structured interaction with the memory system through reflection
commands, supporting memory search, traversal, and relationship exploration.

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

class ReflectionEnvironment(BaseEnvironment):
    """
    Environment for reflecting on and exploring memories.
    Provides commands for memory search, recent recall, and connection traversal.
    """
    
    def __init__(self, cognitive_bridge: CognitiveBridge):
        self.cognitive_bridge = cognitive_bridge
        super().__init__()
        
        # Extended help text for the environment
        self.help_text = """
        The reflection environment lets you explore and reflect on memories.
        Use 'query' to search for specific memories or themes.
        Use 'recent' to review recent experiences.
        
        Examples:
        - reflect query "happy memories"
        - reflect query "conversations about projects"
        - reflect recent --limit=3
        """

    def _register_commands(self) -> None:
        """Register reflection environment commands."""
        self.register_command(
            CommandDefinition(
                name="query",
                description="Reflect on memories matching a topic or theme",
                parameters=[
                    Parameter(
                        name="topic",
                        description="Topic or theme to reflect on",
                        required=True,
                        examples=[
                            '"happy memories"',
                            '"conversations with user"',
                            '"times I learned something"'
                        ]
                    )
                ],
                flags=[
                    Flag(
                        name="depth",
                        description="How deeply to traverse memory connections",
                        type=ParameterType.INTEGER,
                        default=1
                    )
                ],
                examples=[
                    'reflect query "times I felt proud"',
                    'reflect query "recent conversations" --depth=2',
                    'reflect query "moments of insight"'
                ],
                related_commands=[
                    'reflect recent',
                    'meditate focus "related feeling"'
                ],
                category="Memory Search"
            )
        )

        self.register_command(
            CommandDefinition(
                name="recent",
                description="Review recent memory nodes",
                parameters=[],
                flags=[
                    Flag(
                        name="limit",
                        description="Number of recent memories to retrieve",
                        type=ParameterType.INTEGER,
                        default=5
                    )
                ],
                examples=[
                    'reflect recent',
                    'reflect recent --limit=3'
                ],
                related_commands=[
                    'reflect query "recent topic"'
                ],
                category="Memory Review"
            )
        )
        
    async def _execute_command(
        self,
        action: str,
        params: List[Any],
        flags: Dict[str, Any],
        context: Dict[str, Any]
    ) -> CommandResult:
        """Execute reflection commands."""
        try:
            if action == "query":
                topic = params[0]
                depth = flags.get("depth", 1)
                
                # Get relevant memories through cognitive bridge
                nodes = await self.cognitive_bridge.reflect_on_topic(topic, depth)
                
                if not nodes:
                    return CommandResult(
                        success=True,
                        message="No specific memories surfaced for this topic.",
                        suggested_commands=[
                            'reflect query "similar topics"',
                            'reflect recent'
                        ]
                    )
                
                # Format memory descriptions
                descriptions = []
                for node in nodes:
                    descriptions.append(f"• {node.text_content}")
                    # If we traversed connections, show relationship
                    if depth > 1 and node.connections:
                        descriptions.append(
                            "  Connected to: " + 
                            ", ".join(f"{id}" for id in list(node.connections.keys())[:2])
                        )
                
                response = (
                    f"Reflecting on '{topic}':\n\n" +
                    "\n".join(descriptions) +
                    f"\n\nFound {len(nodes)} related memories."
                )
                
                # Suggest relevant follow-ups
                suggested = [
                    f'reflect query "{topic}" --depth={depth + 1}',
                    f'meditate focus "{topic}"'
                ]
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "topic": topic,
                        "depth": depth,
                        "node_count": len(nodes),
                        "node_ids": [n.node_id for n in nodes]
                    }
                )
                
            elif action == "recent":
                limit = flags.get("limit", 5)
                
                # Get recent memories
                recent_nodes = await self.cognitive_bridge.get_recent_memories(limit)
                
                if not recent_nodes:
                    return CommandResult(
                        success=True,
                        message="No recent memories found.",
                        suggested_commands=['reflect query "any topic"']
                    )
                
                # Format recent memory descriptions
                descriptions = []
                for node in recent_nodes:
                    descriptions.append(f"• {node.text_content}")
                
                response = (
                    f"Recent memories:\n\n" +
                    "\n".join(descriptions)
                )
                
                # Suggest relevant follow-ups based on content
                suggested = ['reflect query "expand on recent"']
                if len(recent_nodes) == limit:
                    suggested.append(f'reflect recent --limit={limit + 3}')
                
                return CommandResult(
                    success=True,
                    message=response,
                    suggested_commands=suggested,
                    data={
                        "limit": limit,
                        "node_count": len(recent_nodes),
                        "node_ids": [n.node_id for n in recent_nodes]
                    }
                )
                
        except Exception as e:
            return CommandResult(
                success=False,
                message=f"Reflection failed: {str(e)}",
                suggested_commands=[
                    'reflect recent',
                    'help'
                ],
                error=str(e)
            )