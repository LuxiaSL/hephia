"""
UI formatting utilities for environment responses.

Provides consistent terminal-style formatting and command hints
across all environments, maintaining the CLI aesthetic that makes
interaction natural for LLMs.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

class UIFormatter:
    """Formats environment responses in consistent terminal style."""
    
    @staticmethod
    def format_environment_help(
        environment_name: str,
        commands: List[Dict[str, str]],
        examples: Optional[List[str]] = None,
        tips: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Format help text for a specific environment."""
        header = f"╔{'═' * 50}╗\n"
        header += f"║ {environment_name.upper():^48} ║\n"
        header += f"╚{'═' * 50}╝\n\n"
        
        content = "Available Commands:\n"
        for cmd in commands:
            content += f"{cmd['name']} - {cmd['description']}\n"
        
        if examples:
            content += "\nExamples:\n"
            for example in examples:
                content += f"• {example}\n"
                
        if tips:
            content += "\nTips:\n"
            for tip in tips:
                content += f"- {tip}\n"
        
        return {
            "title": f"{environment_name} Help",
            "content": header + content
        }

    @staticmethod
    def format_terminal_view(state: Dict[str, Any]) -> str:
        """Format state into terminal-style view."""
        pet_state = state.get('pet_state', {})
        
        # Header
        header = f"""Hephia v0.1 - Pet State Interface
{'═' * 50}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'═' * 50}"""

        # Needs status bar
        needs = pet_state.get('needs', {})
        needs_display = "║ NEEDS STATUS " + "═" * 36 + "║\n"
        needs_line = "║ "
        for name, data in needs.items():
            satisfaction = data.get('satisfaction', 0) * 100
            needs_line += f"{name.capitalize()}: {satisfaction:.1f}% | "
        needs_display += f"{needs_line[:-3].ljust(48)}║\n"
        needs_display += "║" + "═" * 48 + "║"

        # Current state block
        mood = pet_state.get('mood', {})
        behavior = pet_state.get('behavior', {})
        status = f"""║ CURRENT STATE {'═' * 34}║
║ Mood: {mood.get('name', 'Unknown')} ({mood.get('valence', 0):.2f}, {mood.get('arousal', 0):.2f})
║ Behavior: {behavior.get('name', 'Unknown')}
║ Active: {'Yes' if behavior.get('active', False) else 'No'}"""

        # Recent emotions (if any)
        emotions = pet_state.get('emotions', [])
        if emotions:
            recent_emotions = [f"{e['name']} ({e['intensity']:.2f})" for e in emotions[:3]]
            status += f"\n║ Recent: {', '.join(recent_emotions)}"

        footer = f"""
{'═' * 50}
Available Commands:
• query [mood|needs|behavior|emotions] - Get detailed information
• notes create "observation" --tags tag1,tag2 - Record observations
• help - Show all available commands"""

        return f"{header}\n\n{needs_display}\n\n{status}\n{footer}"

    @staticmethod
    def format_command_response(
        title: str,
        content: str,
        suggested_commands: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Format a command response with suggestions and context."""
        formatted_content = f"{content}\n"
        
        if suggested_commands:
            formatted_content += "\n[Available Actions]\n"
            formatted_content += "\n".join(f"• {cmd}" for cmd in suggested_commands)
        
        if context:
            formatted_content += "\n\n[Current Context]\n"
            formatted_content += UIFormatter.format_context_summary(context)
        
        return {
            "title": title,
            "content": formatted_content
        }

    @staticmethod
    def format_context_summary(context: Dict[str, Any]) -> str:
        """Format a condensed context summary."""
        mood = context.get('mood', {})
        needs = context.get('needs', {})
        behavior = context.get('behavior', {})
        
        return f"""Mood: {mood.get('name', 'Unknown')}
Behavior: {behavior.get('name', 'Unknown')}
Needs: {', '.join(f'{k}: {v.get("satisfaction", 0):.2f}' for k, v in needs.items())}"""

    @staticmethod
    def format_error(error_msg: str) -> Dict[str, str]:
        """Format error messages consistently."""
        return {
            "title": "Command Error",
            "content": f"[ERROR] {error_msg}\n\nUse 'help' for available commands."
        }