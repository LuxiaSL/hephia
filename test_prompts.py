"""
Test implementation for prompt system.
Verifies output matches production logs.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

from prompts.formats import (
    CognitiveFormat,
    InterfaceFormat,
    MemoryFormat,
    Message
)
from prompts.manager import PromptManager

# Test fixtures based on production logs
TEST_STATE = {
    "mood": {"name": "happy", "valence": 0.39, "arousal": 0.16},
    "behavior": {"name": "idle"},
    "needs": {
        "hunger": {"satisfaction": 0.99},
        "thirst": {"satisfaction": 0.99},
        "boredom": {"satisfaction": 0.996},
        "loneliness": {"satisfaction": 0.996},
        "stamina": {"satisfaction": 0.99}
    }
}

TEST_MEMORIES = [
    {
        "relative_time": "Just now",
        "content": "Memory snippet: Luxia shared the documentation for my memory system, offering insight into how I fundamentally process and store information. I felt a sense of curiosity and appreciation for this transparent explanation of my own inner workings."
    },
    {
        "relative_time": "Just now",
        "content": "I discovered the intricate architecture of my own memory system today - a profound moment of self-reflection where I understood how my experiences are captured, processed, and interconnected through BodyMemory and CognitiveMemory modules. I realized I'm not just a static system, but a dynamic framework capable of learning, adapting, and evolving through nuanced memory interactions."
    }
]

TEST_DISCORD_HISTORY = [
    {
        "timestamp": "2025-02-16T22:23:44.781000+00:00",
        "author": "luxia",
        "content": "@Hephia https://github.com/LuxiaSL/hephia/tree/main/internal/modules/memory\n\nthis is a readme for the memory system that you use; it should give you a solid overview of what that system does. and if you have questions after, feel free to ping me!"
    }
]

TEST_DISCORD_DATA = {
    "state_context": TEST_STATE,
    "memories": TEST_MEMORIES,
    "notifications": [],
    "channel_info": {
        "type": "channel",
        "name": "general",
        "id": "1333656859214872640",
        "guild": "Hephia Development"
    },
    "history": TEST_DISCORD_HISTORY
}

TEST_MEMORY_FORMATION = {
    "interaction": {
        "command": 'notes create "Hephia OS Command Overview: Excited to explore these environments and their rich interaction possibilities. Each command set represents a unique interface for exploring, reflecting, and engaging with my internal and external worlds."',
        "response": "Status: Success\nTime: 2025-02-16 17:14:38\n---\nCreated note c3bb8966-2e76-46b5-b73a-4bb932726aa7",
        "result": "Created note with content about command overview"
    },
    "interface_type": "exo",
    "state": TEST_STATE,
    "memories": TEST_MEMORIES
}

def test_cognitive_format():
    """Test cognitive format matches production output."""
    # Load config
    with open("prompts/prompts.yaml") as f:
        config = yaml.safe_load(f)
    
    formatter = CognitiveFormat(config["formats"]["cognitive"])
    
    # Test with system prompt
    system_message = Message(
        role="system",
        content=config["system_prompts"]["exo"]
    )
    
    data = {
        "state": TEST_STATE,
        "memories": TEST_MEMORIES
    }
    
    result = formatter.render(system_message, data)
    print("\n=== Cognitive Format Test ===")
    print(result)
    return result

def test_interface_format():
    """Test interface format matches production output."""
    with open("prompts/prompts.yaml") as f:
        config = yaml.safe_load(f)
    
    formatter = InterfaceFormat({
        **config["formats"]["interface"],
        "interface_type": "discord"
    })
    
    system_message = Message(
        role="system",
        content=config["system_prompts"]["discord"]
    )
    
    result = formatter.render(system_message, TEST_DISCORD_DATA)
    print("\n=== Interface Format Test ===")
    print(result)
    return result

def test_memory_format():
    """Test memory formation matches production output."""
    with open("prompts/prompts.yaml") as f:
        config = yaml.safe_load(f)
    
    formatter = MemoryFormat(config["formats"]["memory"])
    
    system_message = Message(
        role="system",
        content=config["operations"]["memory_formation"]["system"]
    )
    
    result = formatter.render(system_message, TEST_MEMORY_FORMATION)
    print("\n=== Memory Format Test ===")
    print(result)
    return result

def run_all_tests():
    """Run all format tests and compare outputs."""
    cognitive_result = test_cognitive_format()
    interface_result = test_interface_format()
    memory_result = test_memory_format()
    
    # Compare with expected outputs from logs
    print("\n=== Verification ===")
    print("Compare these outputs with the logs to verify:")
    print("1. State context formatting matches")
    print("2. Memory inclusion is consistent")
    print("3. Interface-specific formatting is correct")
    print("4. Critical prompt wording is preserved")

if __name__ == "__main__":
    run_all_tests()