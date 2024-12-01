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

    BOX_WIDTH = 80
    INNER_WIDTH = BOX_WIDTH - 4  # Accounting for margins

    @staticmethod
    def format_context_summary(context: Dict[str, Any]) -> str:
        """Format concise state summary."""
        pet_state = context.get('pet_state', {})
        mood = pet_state.get('mood', {})
        needs = pet_state.get('needs', {})
        behavior = pet_state.get('behavior', {})

        need_strings = [
            f"{k}: {v.get('satisfaction', 0):.2f}"
            for k, v in needs.items()
        ]
        needs_formatted = ', '.join(need_strings)
        
        return (
            f"Mood: {mood.get('name', 'Unknown')}\n"
            f"Behavior: {behavior.get('name', 'Unknown')}\n"
            f"Needs: {needs_formatted}"
        )

    @staticmethod
    def format_command_result(result: CommandResult, state: Dict[str, Any]) -> str:
        """Format command execution result with state context."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Start with header
        output = [
            "╔" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╗",
            f"║ Status: {'Success' if result.success else 'Error':<71} ║",
            f"║ Time: {timestamp:<73} ║",
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣",
            ""
        ]

        # Add main content
        output.extend([result.message, ""])

        # Add error information if present
        if not result.success and result.error:
            output.extend([
                "Error Details:",
                f"- Type: {result.error}",
                "- Suggested Fixes:",
                *[f"  • {fix}" for fix in result.error.suggested_fixes],
                ""
            ])

        # Add command suggestions
        if result.suggested_commands:
            output.extend([
                "║ Available Actions:",
                *[f"║ • {cmd}" for cmd in result.suggested_commands],
                ""
            ])

        # Add state information
        output.extend([
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣",
            "║ Current State:",
            TerminalFormatter.format_context_summary(state),
            "╚" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╝"
        ])

        return "\n".join(output)

    @staticmethod
    def format_environment_help(env: EnvironmentCommands) -> CommandResult:
        """Format comprehensive environment help."""
        # Header with environment info
        sections = [
            "╔" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╗",
            f"║ {env.environment.upper():^{TerminalFormatter.BOX_WIDTH-4}} ║",
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣",
            "",
            env.description.strip() if env.description else "",
            ""
        ]

        # Group commands by category
        categorized: Dict[str, List[CommandDefinition]] = {}
        for cmd in env.commands.values():
            category = cmd.category or "General"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(cmd)

        # Format each category
        for category, commands in categorized.items():
            sections.extend([
                f"{category}:",
                "-------------------"
            ])

            for cmd in commands:
                # Command signature
                params = " ".join(
                    f"<{p.name}>" if p.required else f"[{p.name}]"
                    for p in cmd.parameters
                )
                flags = " ".join(
                    f"[--{f.name}]" for f in cmd.flags
                )
                signature = f"{env.environment} {cmd.name} {params} {flags}".strip()
                
                sections.extend([
                    signature,
                    f"  {cmd.description}",
                    ""
                ])

                # Parameter details if any
                if cmd.parameters:
                    sections.append("  Parameters:")
                    for param in cmd.parameters:
                        sections.append(
                            f"    {param.name}: {param.description}"
                            f"{' (required)' if param.required else ''}"
                        )
                    sections.append("")

                # Flag details if any
                if cmd.flags:
                    sections.append("  Flags:")
                    for flag in cmd.flags:
                        sections.append(
                            f"    --{flag.name}: {flag.description}"
                            f" (default: {flag.default})" if flag.default is not None else ""
                        )
                    sections.append("")

                # Examples if any
                if cmd.examples:
                    sections.append("  Examples:")
                    sections.extend(f"    {ex}" for ex in cmd.examples)
                    sections.append("")

            sections.append("")  # Space between categories

        # Generate suggested commands
        suggested = [f"{env.environment} help"]  # Always include help
        for cmd in env.commands.values():
            if cmd.examples:
                suggested.append(cmd.examples[0])  # Add first example from each command
                if len(suggested) >= 5:  # Limit to 5 suggestions
                    break

        return CommandResult(
            success=True,
            message="\n".join(sections),
            suggested_commands=suggested,
            data={"environment": env.environment}
        )

    @staticmethod
    def format_error(error: CommandValidationError) -> str:
        """Format error messages consistently."""
        sections = [
            "╔" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╗",
            f"║ Error: {error.message:<{TerminalFormatter.BOX_WIDTH-10}} ║",
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣",
            ""
        ]

        if error.suggested_fixes:
            sections.extend([
                "Suggested Fixes:",
                *[f"• {fix}" for fix in error.suggested_fixes],
                ""
            ])

        if error.examples:
            sections.extend([
                "Examples:",
                *[f"• {ex}" for ex in error.examples],
                ""
            ])

        if error.related_commands:
            sections.extend([
                "Related Commands:",
                *[f"• {cmd}" for cmd in error.related_commands],
                ""
            ])

        sections.extend([
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣",
            "║ Use 'help' for available commands",
            "╚" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╝"
        ])

        return "\n".join(sections)

    @staticmethod
    def format_welcome() -> str:
        """Format welcome message."""
        return (
            "╔" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╗\n"
            "║" + " " * ((TerminalFormatter.BOX_WIDTH - 24) // 2) +
            "Welcome to Hephia OS" +
            " " * ((TerminalFormatter.BOX_WIDTH - 24) // 2) + "║\n"
            "╠" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╣\n"
            "║ Type 'help' to see available commands" +
            " " * (TerminalFormatter.BOX_WIDTH - 39) + "║\n"
            "╚" + "═" * (TerminalFormatter.BOX_WIDTH - 2) + "╝"
        )