"""
terminal_formatter.py - Terminal output formatting for Hephia's CLI interface.

Handles the presentation of command results, help information, and system state
in a consistent and informative format that guides LLM interaction.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from brain.commands.model import (
    CommandResult,
    CommandDefinition,
    Parameter,
    Flag,
    EnvironmentCommands,
    CommandValidationError
)

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
            f"Experiencing a {mood.get('name', 'neutral')} mood "
            f"(valence: {valence:.2f}, arousal: {arousal:.2f}). "
        )
        
        # Format behavior as active state
        behavior_str = (
            f"Currently in a {behavior.get('name', 'balanced')} behavior"
        )
        
        # Format needs as motivational state  
        needs_lines = []
        for need, details in needs.items():
            if isinstance(details, dict) and 'satisfaction' in details:
                satisfaction = details['satisfaction'] * 100
                urgency = "high" if satisfaction < 30 else "moderate" if satisfaction < 70 else "low"
                needs_lines.append(
                    f"• {need}: {satisfaction:.1f}% satisfied ({urgency} urgency)"
                )
        needs_str = "Current needs and their satisfaction levels:\n" + "\n".join(needs_lines)
        
        # Format emotional state as recent experience
        emotions_str = ""
        if emotional_state:
            recent_emotions = ", ".join(
                f"{e['name']} ({e['intensity']:.2f})" for e in emotional_state
            )
            emotions_str = f"\nFeeling: {recent_emotions}"

        # Build core state string
        state_str = (
            "=== Current Internal State ===\n"
            f"{mood_str}\n\n"
            f"{behavior_str}\n\n"
            f"{needs_str}"
            f"{emotions_str}\n"
            "==============================="
        )

        # Add memories if provided
        if memories:
            memory_lines = ["\nRelevant Memories:", "---"]
            for memory in memories:
                # Extract core memory data
                text = memory.get('content', memory.get('text_content', ''))
                # Get first line or snippet of content 
                snippet = text.split('\n')[0]
                if len(snippet) > 250:
                    snippet = snippet[:247] + "..."
                memory_lines.append(f"• {snippet}")
            memory_lines.append("---")
            
            state_str = state_str + "\n" + "\n".join(memory_lines)

        return state_str
    
    @staticmethod
    def format_notifications(notifications: List[str], result: str) -> str:
        """
        Format notifications and append them to the command result with consistent styling.

        Args:
            notifications: List of notification strings to format
            result: Original command result string to append to

        Returns:
            String combining result and formatted notifications
        """
        if not notifications:
            return result

        # Split result at final divider
        parts = result.rsplit("---", 1)
        if len(parts) != 2:
            return result

        # Format notifications
        notification_lines = ["Notifications:"]
        for notice in notifications:
            notification_lines.append(f"• {notice}")
        notification_lines.append("")

        # Reconstruct with notifications before final divider
        return parts[0] + "\n" + "\n".join(notification_lines) + "---" + parts[1]

    @staticmethod
    def format_command_result(result: CommandResult) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = []
        lines.append(f"Status: {'Success' if result.success else 'Error'}")
        lines.append(f"Time: {timestamp}")
        lines.append("---")

        lines.append(result.message.strip())
        lines.append("")

        if not result.success and result.error:
            lines.append("Error Details:")
            lines.append(f"- Type: {result.error.message}")
            if result.error.suggested_fixes:
                lines.append("Suggested Fixes:")
                for fix in result.error.suggested_fixes:
                    lines.append(f"  • {fix}")
            lines.append("")

        if result.suggested_commands:
            lines.append("Available Actions:")
            for cmd in result.suggested_commands:
                lines.append(f"• {cmd}")
        else:
            lines.append("Type 'help' for available commands. Try different environments to explore their capabilities.")
        lines.append("")

        lines.append("---")

        return "\n".join(lines)

    @staticmethod
    def format_environment_help(env: EnvironmentCommands) -> CommandResult:
        lines = []
        lines.append(env.environment.upper())
        lines.append(env.description.strip() if env.description else "")
        lines.append("---")

        # Group commands by category
        categorized: Dict[str, List[CommandDefinition]] = {}
        for cmd in env.commands.values():
            category = cmd.category or "General"
            categorized.setdefault(category, []).append(cmd)

        # Format each category
        for category, commands in categorized.items():
            lines.append(f"{category}:")
            for c in commands:
                # Command signature
                params = " ".join(f"<{p.name}>" if p.required else f"[{p.name}]" for p in c.parameters)
                flags = " ".join(f"[--{f.name}]" for f in c.flags)
                signature = f"{env.environment} {c.name} {params} {flags}".strip()
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
                lines.append("")

        # Suggested commands
        suggested = [f"{env.environment} help"]
        for c in env.commands.values():
            if c.examples:
                suggested.append(c.examples[0])
                if len(suggested) >= 5:
                    break

        return CommandResult(
            success=True,
            message="\n".join(lines),
            suggested_commands=suggested,
            data={"environment": env.environment}
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
        lines.append("---")
        lines.append("Use 'help' for available commands")
        return "\n".join(lines)

    @staticmethod
    def format_welcome() -> str:
        return (
            "Welcome to Hephia OS\n"
            "Type 'help' to see available commands\n"
            "---"
        )