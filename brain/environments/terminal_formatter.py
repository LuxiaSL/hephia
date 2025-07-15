"""
terminal_formatter.py - Terminal output formatting for Hephia's CLI interface.

Handles the presentation of command results, help information, and system state
in a consistent and informative format that guides LLM interaction.
"""

import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from brain.commands.model import (
    CommandResult,
    CommandDefinition,
    EnvironmentCommands,
    CommandValidationError,
    ParsedCommand
)
from brain.environments.base_environment import BaseEnvironment

from loggers import BrainLogger

class TerminalFormatter:
    """Formats system output in a consistent, terminal-like format."""

    @staticmethod    
    def format_context_summary(context: Dict[str, Any], memories: Optional[List[Dict]] = None) -> str:
        """
        Format the current cognitive and emotional state into a cohesive first-person context
        for LLM embodiment and continuity. Optionally includes relevant memories.

        Args:
            context: Dictionary containing current emotional and cognitive state
            memories: Optional list of memory nodes to include in context

        Returns:
            Formatted string combining state and relevant memories
        """
        # Format core state components
        mood = context.get('mood', {})
        needs = context.get('needs', {})
        behavior = context.get('behavior', {})
        emotional_state = context.get('emotional_state', [])

        # Format mood as embodied experience
        valence = mood.get('valence', 0)
        arousal = mood.get('arousal', 0)
        mood_str = (
            f"{mood.get('name', 'neutral')} mood "
            f"(valence: {valence:.2f}, arousal: {arousal:.2f}). "
        )
        
        # Format behavior as active state
        behavior_str = (
            f"{behavior.get('name', 'idle')} behavior"
        )
        
        # Format needs as motivational state  
        needs_lines = []
        for need, details in needs.items():
            if isinstance(details, dict) and 'satisfaction' in details:
                satisfaction = details['satisfaction'] * 100
                urgency = "high" if satisfaction < 30 else "moderate" if satisfaction < 70 else "low"
                needs_lines.append(
                    f"{need}: {satisfaction:.1f}% satisfied ({urgency} urgency)"
                )
        needs_str = ", ".join(needs_lines)
        
        # Format emotional state as recent experience
        emotions_str = ""
        if emotional_state:

            # Separate overall stimulus from individual vectors
            overall_emotion = None
            individual_emotions = []
            
            #sort by intensity, highest should be primary
            emotional_state.sort(key=lambda e: e.get('intensity', 0), reverse=True)
            
            overall_emotion = emotional_state[0] if emotional_state else None
            for emotion in emotional_state[1:]:
                individual_emotions.append(emotion)

            # Format overall emotion as primary feeling
            overall_str = ""
            if overall_emotion:
                overall_str = f"feeling {overall_emotion['name']} ({overall_emotion['intensity']:.2f})"
            
            # Format individual emotions as nuanced layers
            individual_str = ""
            if individual_emotions:
                emotion_descriptors = []
                for emotion in individual_emotions[:8]:  # Limit to 8 for readability
                    emotion_descriptors.append(f"{emotion['name']} ({emotion['intensity']:.2f})")
                individual_str = ", ".join(emotion_descriptors)
            
            # Combine with natural flow
            if overall_str and individual_str:
                emotions_str = f"{overall_str}, with traces of {individual_str}"
            elif overall_str:
                emotions_str = overall_str
            elif individual_str:
                emotions_str = f"experiencing {individual_str}"
            else:
                emotions_str = "emotionally stable"

        # Build core state string
        state_str = (
            "\nMy Internal State\n"
            f"{mood_str}\n"
            f"{behavior_str}\n"
            f"{needs_str}\n"
            f"{emotions_str}\n"
        )

        # Add memories if provided
        if memories:
            memory_lines = ["\nRelevant Memories:"]
            current_time = time.time()
            
            for memory in memories:
                # Extract timestamp and format relative time
                timestamp = memory.get('timestamp', 0)
                time_diff = current_time - timestamp
                
                if time_diff < 300:  # 5 minutes
                    time_str = "Just now"
                elif time_diff < 3600:  # 1 hour
                    time_str = "Recently"
                elif time_diff < 86400:  # 24 hours
                    time_str = "Earlier today"
                elif time_diff < 604800:  # 1 week
                    time_str = "A few days ago"
                elif time_diff < 2592000:  # 30 days
                    time_str = "A while ago"
                else:
                    time_str = "A long time ago"

                # Extract and format memory content
                text = memory.get('content', memory.get('text_content', ''))
                
                # Format memory entry with timestamp and full content
                memory_lines.append(f"[{time_str}]")
                for line in text.split('\n'):
                    if line.strip():
                        memory_lines.append(f"  {line.strip()}")
                memory_lines.append("")

            # Append all memory lines at once after processing all memories
            state_str = state_str + "\n" + "###\n".join(memory_lines)

        return state_str
    
    @staticmethod
    def format_command_cleanly(command: ParsedCommand) -> str:
        """
        Format a successfully parsed command back into clean canonical syntax.
        
        This reconstructs the clean command that was actually executed,
        not the raw LLM output that may have contained formatting issues.

        Args:
            command: ParsedCommand object containing the executed command details

        Returns:
            Clean canonical command string (e.g., 'notes create "My Note" --tags=work')
        """
        if not isinstance(command, ParsedCommand):
            return "Invalid command structure"

        # Start building the command string
        parts = []
        
        # Add environment if it exists (global commands have None environment)
        if command.environment:
            parts.append(command.environment)
        
        # Add action
        parts.append(command.action)
        
        # Add parameters, properly quoted if they contain spaces
        for param in command.parameters:
            if ' ' in param:
                parts.append(f'"{param}"')
            else:
                parts.append(param)
        
        # Add flags in --name=value format
        for flag_name, flag_value in command.flags.items():
            if ' ' in flag_value:
                parts.append(f'--{flag_name}="{flag_value}"')
            else:
                parts.append(f'--{flag_name}={flag_value}')
        
        return ' '.join(parts)
    
    @staticmethod
    def format_notifications(updates: str, result: str) -> str:
        """
        Format cognitive updates/notifications and append them to the command result.

        Args:
            updates: String containing updates from other cognitive interfaces
            result: Original command result string to append to

        Returns:
            String combining result and formatted cognitive updates
        """
        if not updates or updates.strip() == "No recent updates from other interfaces":
            return result

        # Format updates section
        update_lines = [
            "\nCognitive Updates:",
            "###"
        ]
        
        # Split updates by newlines and format each section
        for update in updates.split("\n\n"):
            if update.strip():
                # Indent update content
                indented_update = "\n".join(
                    f"  {line}" for line in update.split("\n") if line.strip()
                )
                update_lines.append(indented_update)
                update_lines.append("")

        # Split result at final divider if it exists
        parts = result.rsplit("###", 1)
        if len(parts) == 2:
            return parts[0] + "\n" + "\n".join(update_lines) + "\n###" + parts[1]
        else:
            return result + "\n" + "\n".join(update_lines) + "\n###"

    @staticmethod
    def format_command_result(result: CommandResult) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = []
        lines.append(f"Status: {'Success' if result.success else 'Error'}")
        lines.append(f"Time: {timestamp}")

        lines.append(result.message.strip())

        if not result.success and result.error:
            lines.append("Error Details:")
            lines.append(f"- Type: {result.error.message}")
            if result.error.suggested_fixes:
                lines.append("Suggested Fixes:")
                for fix in result.error.suggested_fixes:
                    lines.append(f"  • {fix}")

        if result.suggested_commands:
            lines.append("Available Actions:")
            for cmd in result.suggested_commands:
                lines.append(f"• {cmd}")
        else:
            lines.append("Type 'help' for available commands. Try different environments to explore their capabilities.")
        lines.append("###")

        return "\n".join(lines)

    @staticmethod
    def format_environment_help(env: BaseEnvironment) -> CommandResult:
        lines = []
        lines.append(env.name.upper())
        lines.append(env.description.strip() if env.description else "")

        # Group commands by category
        categorized: Dict[str, List[CommandDefinition]] = {}
        for cmd in env.commands.values():
            category = cmd.category or "Global"
            categorized.setdefault(category, []).append(cmd)

        # Format each category
        for category, commands in categorized.items():
            lines.append(f"{category}:")
            for c in commands:
                # Command signature
                params = " ".join(f"<{p.name}>" if p.required else f"[{p.name}]" for p in c.parameters)
                flags = " ".join(f"[--{f.name}]" for f in c.flags)
                signature = f"{env.name} {c.name} {params} {flags}".strip()
                lines.append(signature)
                lines.append(f"  {c.description}")
                if c.parameters:
                    lines.append("  Parameters:")
                    for p in c.parameters:
                        req = " (required)" if p.required else ""
                        lines.append(f"    {p.name}: {p.description}{req}")
                if c.flags:
                    lines.append("  Flags:")
                    for f_ in c.flags:
                        def_str = f"(default: {f_.default})" if f_.default is not None else ""
                        lines.append(f"    --{f_.name}: {f_.description} {def_str}")
                if c.examples:
                    lines.append("  Examples:")
                    for ex in c.examples:
                        lines.append(f"    {ex}")
                lines.append("###")

        # Suggested commands
        suggested = [f"{env.name} help"]
        for c in env.commands.values():
            if c.examples:
                suggested.append(c.examples[0])
                if len(suggested) >= 5:
                    break

        return CommandResult(
            success=True,
            message="\n".join(lines),
            suggested_commands=suggested,
            data={"environment": env.name}
        )

    @staticmethod
    def format_error(error: CommandValidationError) -> str:
        lines = []
        lines.append(f"Error: {error.message}")
        if error.suggested_fixes:
            lines.append("Suggested Fixes:")
            for fix in error.suggested_fixes:
                lines.append(f"• {fix}")
        if error.examples:
            lines.append("Examples:")
            for ex in error.examples:
                lines.append(f"• {ex}")
        if error.related_commands:
            lines.append("Related Commands:")
            for rc in error.related_commands:
                lines.append(f"• {rc}")
        lines.append("Use 'help' for available commands")
        lines.append("###")
        return "\n".join(lines)