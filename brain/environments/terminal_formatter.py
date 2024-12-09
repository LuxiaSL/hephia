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
    def format_context_summary(context: Dict[str, Any]) -> str:
        pet_state = context.get('pet_state', {})
        mood = pet_state.get('mood', {})
        needs = pet_state.get('needs', {})
        behavior = pet_state.get('behavior', {})

        needs_str = ", ".join(f"{k}: {v.get('satisfaction', 0):.2f}" for k,v in needs.items())
        return f"Mood: {mood.get('name','Unknown')}\nBehavior: {behavior.get('name','Unknown')}\nNeeds: {needs_str}"
    
    @staticmethod
    def format_command_result(result: CommandResult, state: Dict[str, Any]) -> str:
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
        lines.append("Current State:")
        lines.append(TerminalFormatter.format_context_summary(state))
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