"""
commands/preprocessor.py - Processes raw LLM output into structured commands.

Handles the initial parsing and validation of LLM command attempts, providing
rich feedback to help the LLM learn and improve its command usage.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
import json
import os
from dataclasses import dataclass
from loggers import BrainLogger
from api_clients import APIManager
from .model import (
    ParsedCommand, 
    CommandDefinition,
    CommandValidationError,
    GlobalCommands,
    EnvironmentCommands
)

class CommandPreprocessor:
    """
    Processes raw LLM output into structured commands.
    Handles hallucination detection, command parsing, and basic validation.
    """
    
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        self.error_count = 0
        self.max_errors = 3
        
        # Patterns for detecting hallucinated terminal output
        self._terminal_patterns = {
            'prompt': r'^(>|\$|#)\s',
            'timestamp': r'\d{2}:\d{2}(:\d{2})?',
            'headers': r'(Terminal|Output|Response):',
            'decorators': r'(-{3,}|\[.*?\])',
            'system_msg': r'Hephia.*?:',
        }

    async def preprocess_command(
        self, 
        command: str, 
        available_commands: Dict[str, EnvironmentCommands]
    ) -> Tuple[Optional[ParsedCommand], Optional[CommandValidationError]]:
        """
        Process LLM response into a structured command.
        
        Args:
            command: Raw LLM output
            available_commands: Registry of available environment commands
            
        Returns:
            Tuple of (parsed_command, error) - one will always be None
        """
        try: 
            # Handle global commands first
            if command.strip().lower() == GlobalCommands.HELP:
                return ParsedCommand(
                    environment=None,
                    action=GlobalCommands.HELP,
                    parameters=[],
                    flags={},
                    raw_input=command
                ), None
                
            # Extract actual command from potential hallucination
            extracted = self._extract_command(command)
            if not extracted:
                error = CommandValidationError(
                    message="Could not extract valid command from output",
                    suggested_fixes=["Use direct command syntax without formatting"],
                    related_commands=self._get_similar_commands(command, available_commands),
                    examples=self._generate_relevant_examples(command, available_commands)
                )
                BrainLogger.log_command_processing(command, None, str(error))
                return None, error
                
            # Sanitize and validate basic structure
            try:
                sanitized = self._sanitize_command(extracted)
                parsed = self._parse_command(sanitized)
            except ValueError as e:
                error = CommandValidationError(
                    message=str(e),
                    suggested_fixes=["Remove special characters", "Use standard command format"],
                    related_commands=[],
                    examples=self._generate_format_examples(available_commands)
                )
                BrainLogger.log_command_processing(command, None, str(error))
                return None, error
                
            # Validate against available commands
            if not self._validate_command(parsed, available_commands):
                # Try to correct command first
                corrected = await self._attempt_command_correction(
                    parsed, available_commands
                )
                if corrected:
                    BrainLogger.log_command_processing(command, corrected, None)
                    return corrected, None
                    
                # If correction fails, provide detailed error
                error = CommandValidationError(
                    message=f"Invalid command: {parsed.raw_input}",
                    suggested_fixes=self._generate_helpful_hints(parsed, available_commands),
                    related_commands=self._get_similar_commands(
                        f"{parsed.environment} {parsed.action}" if parsed.environment else parsed.action,
                        available_commands
                    ),
                    examples=self._generate_relevant_examples(parsed.raw_input, available_commands)
                )
                BrainLogger.log_command_processing(command, None, str(error))
                return None, error
            
            BrainLogger.log_command_processing(command, parsed, None)
            return parsed, None
            
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}"
            BrainLogger.log_command_processing(command, None, error_msg)
            return None, CommandValidationError(
                message=error_msg,
                suggested_fixes=["Try another command format"],
                related_commands=[],
                examples=[]
            )

    def _extract_command(self, text: str) -> Optional[str]:
        """Extract actual command from potentially hallucinated output."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None
            
        first_line = lines[0]
        remaining_lines = lines[1:]
        
        # Handle multi-line commands for specific environments
        if any(first_line.startswith(prefix) for prefix in ('exo query', 'notes create')):
            remaining_content = ' '.join(remaining_lines)
            return f"{first_line} {remaining_content}"
            
        # For single-line commands
        if len(lines) == 1:
            return first_line
            
        # For multi-line input
        command = first_line
        additional_content = ' '.join(
            line for line in remaining_lines 
            if not self._detect_hallucination(line)
        ).strip()
        
        if additional_content:
            if '"' in command:
                # Insert content before last quote
                last_quote = command.rindex('"')
                return f"{command[:last_quote]} {additional_content}{command[last_quote:]}"
            return f"{command} {additional_content}"
            
        return command

    def _parse_command(self, command: str) -> ParsedCommand:
        """Parse command string into structured format."""
        parts = command.split(maxsplit=1)
        if not parts:
            raise ValueError("Empty command")
            
        environment_or_global = parts[0].lower()
        remaining = parts[1] if len(parts) > 1 else ""
        
        # Handle global commands
        if GlobalCommands.is_global_command(environment_or_global):
            return ParsedCommand(
                environment=None,
                action=environment_or_global,
                parameters=[],
                flags={},
                raw_input=command
            )
            
        # Parse environment commands
        if not remaining:
            raise ValueError("Incomplete command")
            
        action_parts = remaining.split(maxsplit=1)
        action = action_parts[0].lower()
        params_str = action_parts[1] if len(action_parts) > 1 else ""
        
        # Parse parameters and flags
        params, flags = self._parse_params_and_flags(params_str)
        
        return ParsedCommand(
            environment=environment_or_global,
            action=action,
            parameters=params,
            flags=flags,
            raw_input=command
        )

    def _parse_params_and_flags(self, params_str: str) -> Tuple[List[str], Dict[str, str]]:
        """Parse parameter string into params and flags."""
        params = []
        flags = {}
        current_param = []
        
        for part in params_str.split():
            if part.startswith("--"):
                if "=" in part:
                    flag_name, flag_value = part[2:].split("=", 1)
                    flags[flag_name] = flag_value
            else:
                if part.startswith('"') and not part.endswith('"'):
                    current_param.append(part)
                elif current_param:
                    current_param.append(part)
                    if part.endswith('"'):
                        params.append(" ".join(current_param)[1:-1])
                        current_param = []
                else:
                    params.append(part)
                    
        return params, flags

    def _validate_command(
        self, 
        command: ParsedCommand, 
        available_commands: Dict[str, EnvironmentCommands]
    ) -> bool:
        """Validate parsed command against available commands."""
        if not command.environment:  # Global command
            return GlobalCommands.is_global_command(command.action)
            
        if command.environment not in available_commands:
            return False
            
        env_commands = available_commands[command.environment].commands
        return command.action in env_commands
    
    async def _attempt_command_correction(
        self, 
        command: ParsedCommand,
        available_commands: Dict[str, EnvironmentCommands]
    ) -> Optional[ParsedCommand]:
        """Attempt to correct invalid command using LLM."""
        try:
            # Format available commands for LLM
            command_help = {}
            for env_name, env in available_commands.items():
                command_help[env_name] = {
                    "description": env.description,
                    "commands": {
                        name: {
                            "description": cmd.description,
                            "parameters": [p.name for p in cmd.parameters],
                            "flags": [f.name for f in cmd.flags],
                            "examples": cmd.examples
                        }
                        for name, cmd in env.commands.items()
                    }
                }
            
            response = await self.api.openrouter.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a command preprocessor for an OS simulator. Your task is to correct invalid commands and provide helpful feedback.
Available commands: {command_help}

Rules:
1. If a command is missing its environment prefix (e.g., 'create' instead of 'notes create'), add the correct prefix
2. If the syntax is incorrect, correct it (e.g., 'notes --"this is an example"' becomes 'notes create "this is an example"')
3. Return a JSON object with two fields: "command" (the corrected command) and "explanation" (what was fixed)
4. Keep the command clean, but maintain any necessary information"""
                    },
                    {
                        "role": "user",
                        "content": f'Command: "{command.raw_input}"'
                    }
                ],
                model=os.getenv("OPENROUTER_MODEL"),
                temperature=0.2
            )

            result = response["choices"][0]["message"]["content"]
            try:
                # Parse JSON response
                correction = json.loads(result)
                corrected_command = correction["command"]
                
                # Parse and validate the corrected command
                parsed = self._parse_command(corrected_command)
                if self._validate_command(parsed, available_commands):
                    return parsed
                    
            except (json.JSONDecodeError, KeyError):
                BrainLogger.log_command_processing(command, None, "Invalid correction format from LLM")
                
        except Exception as e:
            BrainLogger.log_command_processing(command, None, f"Error in command correction: {e}")
            
        return None

    def _sanitize_command(self, command: str) -> str:
        """
        Sanitize command string and check for unsafe patterns.
        
        Args:
            command: Raw command string
            
        Returns:
            Sanitized command string
            
        Raises:
            ValueError: If command contains unsafe patterns
        """
        # Remove ANSI escape sequences
        command = re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', command)
        
        # Remove common terminal artifacts
        command = re.sub(r'^\s*[$>]\s*', '', command)
        
        # Don't strip quotes that are part of command parameters
        if (command.startswith('"') and command.endswith('"') and 
            not ('" --' in command or '" "' in command)):
            command = command[1:-1]
            
        # Check for dangerous patterns
        dangerous_patterns = [
            r'`.*`',  # Backticks
            r'\$\(.+\)',  # Command substitution
            r';\s*\w+',  # Command chain attempts
            r'\|\s*\w+',  # Pipe attempts
            r'>\s*\w+',  # Redirection attempts
            r'<\s*\w+',  # Input redirection
            r'&\s*\w+',  # Background execution
        ]
        
        if any(re.search(pattern, command) for pattern in dangerous_patterns):
            raise ValueError(
                "Command contains potentially unsafe patterns. "
                "Please use simple command structures without special characters."
            )
            
        return command.strip()

    def _detect_hallucination(self, line: str) -> bool:
        """
        Detect if a line appears to be hallucinated terminal output.
        
        Args:
            line: Single line of text to check
            
        Returns:
            True if line appears to be hallucinated output
        """
        # Check each pattern category
        for pattern in self._terminal_patterns.values():
            if re.search(pattern, line):
                return True
                
        # Additional contextual checks
        terminal_indicators = [
            line.startswith('$'),
            line.startswith('>'),
            line.startswith('Terminal:'),
            line.startswith('Output:'),
            bool(re.match(r'\[\d{2}:\d{2}(:\d{2})?\]', line)),  # Timestamp format
            bool(re.match(r'-{3,}', line))  # Horizontal rules
        ]
        
        return any(terminal_indicators)

    def _get_similar_commands(
        self,
        partial: str,
        available_commands: Dict[str, EnvironmentCommands]
    ) -> List[str]:
        """
        Find similar commands based on partial input.
        
        Args:
            partial: Partial command text
            available_commands: Registry of available commands
            
        Returns:
            List of similar command strings
        """
        similars = []
        words = partial.lower().split()
        
        # Look for environment matches
        if len(words) == 1:
            for env_name, env in available_commands.items():
                if env_name.startswith(words[0]):
                    # Add most common commands from this environment
                    for cmd_name, cmd in list(env.commands.items())[:2]:
                        similars.append(f"{env_name} {cmd_name}")
                        
        # Look for command matches within environment
        elif len(words) >= 2:
            env_name = words[0]
            cmd_partial = words[1]
            
            if env_name in available_commands:
                env = available_commands[env_name]
                for cmd_name, cmd in env.commands.items():
                    # Check command name similarity
                    if (cmd_name.startswith(cmd_partial) or 
                        any(word in cmd_name for word in cmd_partial.split('_'))):
                        similars.append(f"{env_name} {cmd_name}")
                        
        return similars[:5]  # Return top 5 most similar

    def _generate_format_examples(
        self, 
        available_commands: Dict[str, EnvironmentCommands]
    ) -> List[str]:
        """
        Generate basic format examples for common command patterns.
        
        Args:
            available_commands: Registry of available commands
            
        Returns:
            List of example command strings
        """
        examples = [
            "help  # Show all available commands",
            "notes help  # Show notes environment commands",
            'notes create "My first note" --tags=important',
            'search query "What is Python?"',
            'web open "https://example.com"'
        ]
        
        # Add some environment-specific examples
        for env_name, env in available_commands.items():
            # Get first command with parameters
            for cmd in env.commands.values():
                if cmd.parameters:
                    param_str = " ".join(f"<{p.name}>" for p in cmd.parameters if p.required)
                    flag_str = " ".join(f"[--{f.name}]" for f in cmd.flags)
                    examples.append(f"# Format for {env_name} {cmd.name}:")
                    examples.append(f"{env_name} {cmd.name} {param_str} {flag_str}")
                    break
                    
        return examples

    def _format_error_message(self, error: str, command: str) -> str:
        """
        Format error messages to be helpful for the LLM.
        
        Args:
            error: Raw error message
            command: Original command that caused the error
            
        Returns:
            Formatted error message
        """
        if "unsafe patterns" in error:
            return (
                f"Command '{command}' contains unsafe patterns. "
                "Please use simple command structures without special characters or operators. "
                "Example: notes create \"My note\" --tags=important"
            )
            
        if "incomplete command" in error.lower():
            return (
                f"Command '{command}' is incomplete. "
                "Most commands need both an environment and an action. "
                "Example: notes create \"Note title\""
            )
            
        # Generic format improvement
        return (
            f"Invalid command structure: {error}. "
            "Please use the format: <environment> <action> [parameters] [--flags]. "
            "Use 'help' to see available commands."
        )

    def _generate_helpful_hints(
        self, 
        command: ParsedCommand,
        available_commands: Dict[str, EnvironmentCommands]
    ) -> List[str]:
        """Generate context-aware helpful hints."""
        hints = []
        
        # Check for environment typos/missing prefix
        if not command.environment:
            for env_name, env in available_commands.items():
                if command.action in env.commands:
                    hints.append(f"Did you mean '{env_name} {command.action}'?")
                    break
                    
        # Check for action typos
        elif command.environment in available_commands:
            env = available_commands[command.environment]
            similar = [
                name for name in env.commands.keys()
                if name.startswith(command.action[:2])
            ]
            if similar:
                hints.extend(f"Did you mean '{command.environment} {name}'?" for name in similar)
                
        # Add general format hint
        if command.environment and command.environment in available_commands:
            env = available_commands[command.environment]
            cmd = env.commands.get(command.action)
            if cmd:
                param_str = " ".join(f"<{p.name}>" for p in cmd.parameters if p.required)
                flag_str = " ".join(f"[--{f.name}]" for f in cmd.flags)
                hints.append(f"Format: {command.environment} {command.action} {param_str} {flag_str}")
                
        return hints

    def _generate_relevant_examples(
        self,
        command: str,
        available_commands: Dict[str, EnvironmentCommands]
    ) -> List[str]:
        """Generate relevant command examples based on context."""
        examples = []
        
        # Find most relevant environment
        relevant_env = None
        if " " in command:
            env_name = command.split()[0]
            if env_name in available_commands:
                relevant_env = available_commands[env_name]
                
        if relevant_env:
            # Add examples from the relevant environment
            for cmd in relevant_env.commands.values():
                examples.extend(cmd.examples[:2])  # First two examples per command
        else:
            # Add general examples from each environment
            for env in available_commands.values():
                if env.commands:
                    cmd = next(iter(env.commands.values()))
                    if cmd.examples:
                        examples.append(cmd.examples[0])
                        
        return examples[:5]  # Return top 5 most relevant examples