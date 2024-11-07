"""
Command preprocessing and validation for Hephia's exo loop.
"""

from typing import Dict, List, Optional, Tuple
import re
import logging
import os
from api_clients import APIManager

logger = logging.getLogger('hephia.command_processor')

class CommandPreprocessor:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        self.error_count = 0
        self.max_errors = 3

    def _detect_hallucination(self, text: str) -> bool:
        """Detect hallucinated terminal output patterns."""
        terminal_patterns = [
            r'Hephia.*?:',
            r'\[.*?\]',
            r'\$\s*$',
            r'>>>',
            r'Terminal:',
            r'Output:',
            r'Response:',
            r'\d{2}:\d{2}(:\d{2})?',  # Time patterns
            r'^(>|\$|#)\s',  # Command prompts
            r'---+'  # Horizontal rules
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in terminal_patterns)

    async def preprocess_command(self, 
                               command: str, 
                               available_commands: Dict[str, List[Dict]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Process LLM response to extract and validate commands.
        Returns (processed_command, help_text) or (None, error_message)
        """
        try:
            logger.debug(f"Processing command: {command}")
            # Extract potential command from response
            extracted = self._extract_command(command)
            if not extracted:
                hint = self._generate_helpful_hint(command, available_commands)
                return None, f"Could not extract valid command. {hint}"

            logger.debug(f"Extracted command: {extracted}")

            # Sanitize command
            try:
                sanitized = self._sanitize_command(extracted)
                logger.debug(f"Sanitized command: {sanitized}")
            except ValueError as e:
                return None, f"Invalid command structure: {str(e)}"

            # Check if command is valid
            if self._is_valid_command(sanitized, available_commands):
                self.error_count = 0  # Reset error count on success
                return sanitized, None

            hint = self._generate_helpful_hint(sanitized, available_commands)

            # If invalid, try to correct with LLM
            corrected = await self._correct_command(command, available_commands)
            if corrected:
                return corrected, f"Corrected command. Original attempt: '{sanitized}'"
            
            self.error_count += 1
            if self.error_count >= self.max_errors:
                return None, "Too many command errors - resetting conversation"

            return None, f"Invalid command: '{sanitized}'. {hint}"
        
        except Exception as e:
            logger.error(f"Error in command preprocessing: {e}")
            return None, f"Error processing command: {str(e)}"

    def _extract_command(self, text: str) -> Optional[str]:
        """Extract actual command from potentially hallucinated output."""
        # Handle direct commands first
        if text.strip().lower() == "help":
            return "help"

        # Split into lines and filter out empty ones
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None

        first_line = lines[0]
        remaining_lines = lines[1:]

        # Handle multi-line commands for specific environments
        if first_line.startswith(('exo query', 'notes create')):
            remaining_content = ' '.join(
                line for line in remaining_lines 
                if not self._detect_hallucination(line)
            )
            if remaining_content:
                if '"' in first_line:
                    # Insert additional content before closing quote
                    last_quote = first_line.rindex('"')
                    return f"{first_line[:last_quote]} {remaining_content}{first_line[last_quote:]}"
                else:
                    return f"{first_line} {remaining_content}"
            return first_line

        # For single-line commands
        if not remaining_lines:
            return self._sanitize_command(first_line)

        # For multi-line input
        command = first_line
        additional_content = ' '.join(
            line for line in remaining_lines 
            if not self._detect_hallucination(line)
        )

        if additional_content:
            if '"' in command:
                last_quote = command.rindex('"')
                return f"{command[:last_quote]} {additional_content}{command[last_quote:]}"
            return f"{command} {additional_content}"

        return self._sanitize_command(command)
    
    def _sanitize_command(self, command: str) -> str:
        """Sanitize and validate command structure."""
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
        ]
        
        if any(re.search(pattern, command) for pattern in dangerous_patterns):
            raise ValueError("Command contains potentially unsafe patterns")
            
        return command.strip()
    
    def _generate_helpful_hint(self, command: str, available_commands: Dict) -> str:
        """Generate helpful hints for invalid commands."""
        first_word = command.split(' ')[0] if command else ''
        
        # Check for missing environment prefix
        if first_word in ['query', 'think', 'explore']:
            return f"Try adding 'exo' prefix: 'exo {command}'"
            
        if first_word in ['create', 'list', 'read', 'update', 'delete', 'tags']:
            return f"Try adding 'notes' prefix: 'notes {command}'"
            
        # Check for web-like commands
        if 'http' in command or 'www' in command:
            return f"Try using 'web open' to visit URLs: 'web open {command}'"
            
        # Check for search-like queries
        if '?' in command or len(command) > 20:
            return f"Try using 'search query' or 'exo query': 'search query {command}'"
            
        # Show available commands
        example_commands = [
            f"{env} {cmd['name']}"
            for env, cmds in available_commands.items()
            for cmd in cmds[:2]  # Just first two commands per environment
        ][:3]  # Only show three total examples
        
        return f"Available command examples: {', '.join(example_commands)}"

    def _is_valid_command(self, command: str, available_commands: Dict[str, List[Dict]]) -> bool:
        """Validate command against available commands."""
        # Handle help command specially
        if command.strip().lower() == "help":
            return True

        parts = command.split()
        if len(parts) < 2:
            return False
            
        env_name, action = parts[0], parts[1]
        if env_name not in available_commands:
            return False
            
        return any(cmd['name'] == action for cmd in available_commands[env_name])

    async def _correct_command(self, command: str, available_commands: Dict[str, List[Dict]]) -> Optional[str]:
        """Use OpenRouter to attempt command correction."""
        try:
            response = await self.api.openrouter.create_completion(
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a command preprocessor for an OS simulator. Your task is to correct invalid commands and provide helpful feedback.

Rules:
1. If a specific command is missing its environment prefix (e.g., 'create' instead of 'notes create'), add the correct prefix
2. If the syntax is incorrect, correct it (e.g., 'notes --"this is an example"' becomes 'notes create "this is an example"')
3. Explain what you did to correct, or any otherwise helpful hint if incorrect.
4. Always return a JSON response in this format: {"processedCommand": "corrected command", "helpText": "explanation"}
5. Maintain any necessary information or specific wording."""
                    },
                    {
                        "role": "user",
                        "content": f"Command: {command}\nValid environments and commands: {available_commands}"
                    }
                ],
                model=os.getenv("OPENROUTER_MODEL"),
                temperature=0.6
            )
            
            corrected = response["choices"][0]["message"]["content"]
            if corrected and self._is_valid_command(corrected, available_commands):
                return corrected
                
        except Exception as e:
            logger.error(f"Error in command correction: {e}")
            
        return None