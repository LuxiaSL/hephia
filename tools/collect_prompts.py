"""
Analyzes prompt logs to extract and compare complete examples of each prompt type.
Focuses on understanding the structure and composition of different prompt templates.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptTemplate:
    """Represents a complete prompt template with all its components."""
    type: str                      # Type identifier for this prompt template
    system_content: str           # The system message content
    user_content: str             # Example user message content
    metadata: Dict[str, Any]      # Associated metadata
    context: Dict[str, Any]       # Contextual information
    message_count: int            # Number of messages in the conversation
    first_seen: datetime          # First occurrence timestamp
    example_flow: List[Dict[str, Any]]  # Complete example conversation flow

class TemplateAnalyzer:
    """Analyzes and extracts prompt templates from logs."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.templates: Dict[str, PromptTemplate] = {}
        
    def load_and_analyze(self) -> None:
        """Load log file and extract templates."""
        with open(self.log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split on boundary markers
        entries = content.split("=" * 80)
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
                
            try:
                # Parse timestamp and JSON content
                lines = entry.split('\n', 2)
                if len(lines) < 3:
                    continue
                    
                json_str = '\n'.join(lines[1:])
                entry_data = json.loads(json_str)
                
                if entry_data and "messages" in entry_data:
                    self._process_entry(entry_data)
                    
            except json.JSONDecodeError:
                continue
    
    def _process_entry(self, entry: Dict[str, Any]) -> None:
        """Process a single log entry to extract template information."""
        messages = entry["messages"]
        
        # Find system message
        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if not system_msg:
            return
            
        # Identify template type
        template_type = self._identify_template_type(system_msg["content"])
        if not template_type:
            return
            
        # Create or update template
        timestamp = datetime.fromisoformat(entry["timestamp"].replace('Z', '+00:00'))
        
        if template_type not in self.templates:
            self.templates[template_type] = PromptTemplate(
                type=template_type,
                system_content=system_msg["content"],
                user_content=messages[1]["content"] if len(messages) > 1 else "",
                metadata=entry["metadata"],
                context={},
                message_count=len(messages),
                first_seen=timestamp,
                example_flow=messages
            )
        elif len(messages) > self.templates[template_type].message_count:
            # Update if we find a more complete example
            self.templates[template_type] = PromptTemplate(
                type=template_type,
                system_content=system_msg["content"],
                user_content=messages[1]["content"] if len(messages) > 1 else "",
                metadata=entry["metadata"],
                context={},
                message_count=len(messages),
                first_seen=self.templates[template_type].first_seen,
                example_flow=messages
            )
    
    def _identify_template_type(self, content: str) -> str:
        """Identify the type of prompt template from system message."""
        identifiers = {
            "memory_formation": "creating autobiographical memory snippets",
            "cognitive_continuity": "maintaining cognitive continuity",
            "base_system": "using and acting as Hephia",
            "discord_interface": "acting as an interface",
            "natural_interaction": "exploring and interacting with both its world",
            "summarization": "Provide a comprehensive summary"
        }
        
        for template_type, identifier in identifiers.items():
            if identifier.lower() in content.lower():
                return template_type
                
        return "unknown"
    
    def print_template_analysis(self) -> None:
        """Print detailed analysis of each template type."""
        print("\n=== Prompt Template Analysis ===\n")
        
        for template_type, template in self.templates.items():
            print(f"Template Type: {template_type}")
            print("=" * 40)
            print("\nSystem Prompt:")
            print("-" * 20)
            print(template.system_content)
            
            print("\nExample User Input:")
            print("-" * 20)
            print(template.user_content)
            
            print("\nMetadata:")
            print("-" * 20)
            for key, value in template.metadata.items():
                print(f"{key}: {value}")
            
            print("\nConversation Structure:")
            print("-" * 20)
            print(f"Message Count: {template.message_count}")
            print("Flow:")
            for msg in template.example_flow:
                print(f"- {msg['role']}")
            
            print("\n" + "=" * 80 + "\n")
    
    def export_templates(self, output_path: Path) -> None:
        """Export templates to a structured format."""
        output = {
            template_type: {
                "system_prompt": template.system_content,
                "example_user_input": template.user_content,
                "metadata": template.metadata,
                "message_count": template.message_count,
                "first_seen": template.first_seen.isoformat(),
                "conversation_flow": [
                    {
                        "role": msg["role"],
                        "content_preview": msg["content"][:100] + "..."
                    }
                    for msg in template.example_flow
                ]
            }
            for template_type, template in self.templates.items()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

def analyze_templates(log_path: str, output_path: Optional[str] = None) -> TemplateAnalyzer:
    """Analyze prompt templates from logs."""
    analyzer = TemplateAnalyzer(Path(log_path))
    analyzer.load_and_analyze()
    analyzer.print_template_analysis()
    
    if output_path:
        analyzer.export_templates(Path(output_path))
    
    return analyzer

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python prompt_analyzer.py <path_to_prompt_log> [output_path]")
        sys.exit(1)
        
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_templates(sys.argv[1], output_path)