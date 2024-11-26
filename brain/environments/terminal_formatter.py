"""
Terminal output formatting for Hephia's CLI interface.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

@dataclass
class CommandResponse:
    """Standard response object from command execution"""
    title: str
    content: str
    suggested_commands: Optional[List[str]] = None

@dataclass
class EnvironmentHelp:
    """Help information for an environment"""
    name: str
    commands: List[Dict[str, str]]
    examples: Optional[List[str]] = None
    tips: Optional[List[str]] = None

class TerminalFormatter:
    @staticmethod
    def format_context_summary(context: Dict[str, Any]) -> str:
        """Format concise state summary"""
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
    def format_response(response: CommandResponse, state: Dict[str, Any]) -> str:
        """Format complete CLI response with command output and state"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        output = (
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║ Command Response: {response.title:<54} ║\n"
            f"║ Time: {timestamp:<63} ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            f"\n{response.content}\n"
        )

        if response.suggested_commands:
            output += "\n║ Available Actions:\n"
            for cmd in response.suggested_commands:
                output += f"║ • {cmd}\n"

        output += (
            "\n╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║ Current State:\n"
            f"{TerminalFormatter.format_context_summary(state)}\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        )
        
        return output
    
    @staticmethod
    def format_environment_help(help_info: EnvironmentHelp) -> CommandResponse:
        """Format help text for a specific environment."""
        header = f"╔{'═' * 50}╗\n"
        header += f"║ {help_info.name.upper():^48} ║\n"
        header += f"╚{'═' * 50}╝\n\n"
        
        content = "Available Commands:\n"
        for cmd in help_info.commands:
            content += f"{cmd['name']} - {cmd['description']}\n"
        
        if help_info.examples:
            content += "\nExamples:\n"
            for example in help_info.examples:
                content += f"• {example}\n"
                
        if help_info.tips:
            content += "\nTips:\n"
            for tip in help_info.tips:
                content += f"- {tip}\n"
        
        return CommandResponse(
            title=f"{help_info.name} Help",
            content=header + content,
            suggested_commands=help_info.examples if help_info.examples else None
        )

    @staticmethod
    def format_error(error_msg: str) -> CommandResponse:
        """Format error messages consistently"""
        return CommandResponse(
            title="Error",
            content=f"[ERROR] {error_msg}\n\nUse 'help' for available commands."
        )

    @staticmethod
    def format_welcome() -> str:
        """Format welcome message"""
        return (
            "╔══════════════════════════════════════════════════════════════════════════════╗\n"
            "║                         Welcome to Hephia OS                                  ║\n"
            "╠══════════════════════════════════════════════════════════════════════════════╣\n"
            "║ Type 'help' to see available commands                                        ║\n"
            "╚══════════════════════════════════════════════════════════════════════════════╝"
        )