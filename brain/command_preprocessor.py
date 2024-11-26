"""
Command preprocessing and validation for Hephia's exo loop.
"""

from typing import Dict, List, Optional, Tuple
import re
import logging
import os
import json
from datetime import datetime
from api_clients import APIManager

logger = logging.getLogger('hephia.command_processor')

class CommandPreprocessor:
    def __init__(self, api_manager: APIManager):
        self.api = api_manager
        self.error_count = 0
        self.max_errors = 3

        # Extended patterns for better hallucination detection
        self._terminal_patterns = {
            'prompt': r'^(>|\$|#)\s',
            'timestamp': r'\d{2}:\d{2}(:\d{2})?',
            'headers': r'(Terminal|Output|Response):',
            'decorators': r'(-{3,}|\[.*?\])',
            'system_msg': r'Hephia.*?:',
        }

    def _detect_hallucination(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Enhanced hallucination detection with pattern identification.
        Returns (is_hallucinated, pattern_type).
        """
        for pattern_type, pattern in self._terminal_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return True, pattern_type
        return False, None

    async def preprocess_command(self, 
                               command: str, 
                               available_commands: Dict[str, List[Dict]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Process LLM response to extract and validate commands.
        Returns (processed_command, help_text) or (None, error_message)
        """
        try:
            logger.debug(f"Processing command: {command}")

            # Check for help command first
            if command.strip().lower() == "help":
                return "help", None
            
            # Extract command, noting any hallucinations
            extracted, hallucination_type = self._extract_command_and_content(command)
            if not extracted:
                hint = self._generate_helpful_hint(command, available_commands, hallucination_type)
                return None, f"Could not extract valid command. {hint}"

           # Sanitize and validate structure
            try:
                sanitized = self._sanitize_command(extracted)
                if sanitized != extracted:
                    logger.debug(f"Sanitized command from '{extracted}' to '{sanitized}'")
            except ValueError as e:
                detailed_error = self._format_error_message(str(e), extracted)
                return None, detailed_error

            # Check if command is valid
            if self._is_valid_command(sanitized, available_commands):
                self.error_count = 0  # Reset error count on success
                return sanitized, None
            
            # Try to correct invalid command
            corrected, correction_type = await self._attempt_command_correction(
                sanitized, available_commands
            )

            if corrected:
                return corrected, f"Corrected command ({correction_type}). Original: '{sanitized}'"

            # Handle repeated errors
            self.error_count += 1
            if self.error_count >= self.max_errors:
                return None, "Too many command errors - resetting conversation for a fresh start"

            hint = self._generate_helpful_hint(sanitized, available_commands)
            return None, f"Invalid command structure: '{sanitized}'. {hint}"
        
        except Exception as e:
            logger.error(f"Error in command preprocessing: {e}")
            return None, f"Error processing command: {str(e)}"

    def _extract_command_and_content(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Enhanced command extraction with content preservation.
        Returns (extracted_command, hallucination_type if any).
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return None, "empty_input"

        first_line = lines[0]
        remaining_lines = lines[1:]

        # Check first line for hallucinations
        is_hallucinated, hall_type = self._detect_hallucination(first_line)
        if is_hallucinated:
            return None, hall_type

        # Handle multi-line commands for specific environments
        if first_line.startswith(('exo query', 'notes create')):
            return self._handle_multiline_command(first_line, remaining_lines)

        # Handle single line commands
        if not remaining_lines:
            return first_line, None

        # Process additional content
        clean_content = self._clean_additional_content(remaining_lines)
        if not clean_content:
            return first_line, None

        return self._merge_command_and_content(first_line, clean_content), None
    
    def _handle_multiline_command(self, command: str, extra_lines: List[str]) -> Tuple[str, Optional[str]]:
        """Handle commands that can accept multiple lines of content."""
        clean_content = self._clean_additional_content(extra_lines)
        if not clean_content:
            return command, None

        if '"' in command:
            last_quote = command.rindex('"')
            return f"{command[:last_quote]} {clean_content}{command[last_quote:]}", None
        return f"{command} {clean_content}", None
    
    def _clean_additional_content(self, lines: List[str]) -> Optional[str]:
        """Clean and combine additional content lines."""
        clean_lines = []
        for line in lines:
            is_hall, _ = self._detect_hallucination(line)
            if not is_hall:
                clean_lines.append(line)
        return ' '.join(clean_lines) if clean_lines else None
    
    def _merge_command_and_content(self, command: str, content: str) -> str:
        """Merge command with additional content preserving structure."""
        if '"' in command:
            last_quote = command.rindex('"')
            return f"{command[:last_quote]} {content}{command[last_quote:]}"
        return f"{command} {content}"
    
    def _format_error_message(self, error: str, command: str) -> str:
        """Format detailed error messages for LLM guidance."""
        if "unsafe patterns" in error:
            return (
                f"Command '{command}' contains unsafe patterns. "
                "Please use simple command structures without special characters or operators."
            )
        return f"Invalid command structure: {error}. Please use standard command formats."
    
    async def _attempt_command_correction(
        self, command: str, available_commands: Dict[str, List[Dict]]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Attempt to correct invalid commands using multiple strategies.
        Returns (corrected_command, correction_type) if successful.
        """
        # Try LLM correction
        llm_correction = await self._correct_command(command, available_commands)
        if llm_correction and self._is_valid_command(llm_correction, available_commands):
            return llm_correction, "llm_correction"

        return None, None
    
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
    
    def _generate_helpful_hint(
        self, command: str, available_commands: Dict, hallucination_type: Optional[str] = None
    ) -> str:
        """Generate context-aware helpful hints."""
        if hallucination_type:
            return (
                f"Detected terminal-like output ({hallucination_type}). "
                "Please provide just the command without formatting."
            )

        first_word = command.split(' ')[0] if command else ''
        
        # Environment-specific hints
        env_hints = {
            ('google', 'query'): ('search', 'search query "your question"'),
            ('create', 'list', 'read', 'update', 'delete', 'tags'): 
                ('notes', 'notes create "your note"'),
            ('open', 'browse'):('web', 'web open "url"')
        }
        
        for words, (env, example) in env_hints.items():
            if first_word in words:
                return f"Did you mean to use the '{env}' environment? Example: {example}"

        # Web-specific hint
        if 'http' in command or 'www' in command:
            return f"To visit URLs, use: 'web open {command}'"

        # Search-like query hint
        if '?' in command or len(command) > 20:
            return (
                "This looks like a search query. Try:\n"
                f"- 'search query {command}' for web search\n"
            )

        # Show available commands
        example_commands = [
            f"{env} {cmd['name']}"
            for env, cmds in available_commands.items()
            for cmd in cmds[:2]
        ][:3]

        return (
            "Available command examples:\n" + 
            "\n".join(f"• {cmd}" for cmd in example_commands)
        )

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
                        "content": """You are a command preprocessor for an OS simulator. Your task is to correct invalid commands and provide helpful feedback.

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
                temperature=0.5
            )
            
            
            corrected = response["choices"][0]["message"]["content"]
            if corrected and self._is_valid_command(corrected, available_commands):
                return corrected
                
        except Exception as e:
            logger.error(f"Error in command correction: {e}")
            
        return None