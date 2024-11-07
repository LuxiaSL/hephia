"""
Comprehensive test suite for Hephia system.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import os
import json
from datetime import datetime
import asyncio

from core.server import HephiaServer
from brain.command_preprocessor import CommandPreprocessor
from api_clients import APIManager
from brain.environments.environment_registry import EnvironmentRegistry

class MockAPIManager:
    """Mock API manager for testing."""
    def __init__(self):
        self.openpipe = AsyncMock()
        self.openrouter = AsyncMock()
        self.perplexity = AsyncMock()
        
        # Setup default responses
        self.openpipe.create_completion.return_value = {
            "choices": [{"message": {"content": "I observe the pet is in a happy state. notes create \"Pet showing positive mood\" --tags observation,mood"}}]
        }
        
        self.openrouter.create_completion.return_value = {
            "choices": [{"message": {"content": "notes create \"Pet showing positive mood\" --tags observation,mood"}}]
        }

@pytest.fixture
async def server():
    """
    Fixture providing a test server instance.
    Uses context manager to ensure proper cleanup.
    """
    with patch.dict(os.environ, {
        'OPENPIPE_API_KEY': 'test_key',
        'OPENPIPE_MODEL': 'test-model',
        'OPENROUTER_API_KEY': 'test_key',
        'OPENROUTER_MODEL': 'test-model',
        'PERPLEXITY_API_KEY': 'test_key'
    }):
        test_server = HephiaServer()
        mock_api = MockAPIManager()
        test_server.api = mock_api
        
        # Initialize server
        await test_server.startup()
        
        try:
            yield test_server  # This is key - yield the actual server instance
        finally:
            # Cleanup
            await test_server.shutdown()

@pytest.mark.asyncio
async def test_server_initialization(server):
    """Test server initializes all components properly."""
    assert server.pet is not None
    assert server.state_bridge is not None
    assert server.api is not None
    assert isinstance(server.api.openpipe, AsyncMock)
    assert server.command_preprocessor is not None
    assert server.environment_registry is not None

@pytest.mark.asyncio
async def test_exo_loop_basic_flow(server):
    """Test complete exo loop flow."""
    # Run one iteration
    await server.process_exo_loop()
    
    # Verify OpenPipe was called
    assert server.api.openpipe.create_completion.called
    
    # Verify command was processed
    args = server.api.openpipe.create_completion.call_args
    assert "messages" in args[1]
    assert any("system" in msg["role"] for msg in args[1]["messages"])

@pytest.mark.asyncio
async def test_command_preprocessing(server):
    """Test command preprocessing with mock APIs."""
    preprocessor = server.command_preprocessor
    
    # Test direct help command
    command, help_text = await preprocessor.preprocess_command(
        "help",
        {"notes": [{"name": "create"}]}
    )
    assert command == "help"
    assert help_text is None
    
    # Test valid notes command
    command, help_text = await preprocessor.preprocess_command(
        "notes create \"Test note\" --tags test",
        {"notes": [{"name": "create"}]}
    )
    assert command == 'notes create "Test note" --tags test'
    assert help_text is None

    # Test invalid command
    command, help_text = await preprocessor.preprocess_command(
        "invalid command",
        {"notes": [{"name": "create"}]}
    )
    assert command is None
    assert "Invalid command" in help_text
    
@pytest.mark.asyncio
async def test_state_bridge_functionality(server):
    """Test state bridge updates and retrieval."""
    state = await server.state_bridge.get_current_state()
    assert 'pet_state' in state
    assert 'brain_state' in state
    assert 'last_updated' in state

@pytest.mark.asyncio
async def test_environment_registry(server):
    """Test environment registry and available commands."""
    registry = server.environment_registry
    commands = registry.get_available_commands()
    
    assert 'notes' in commands
    assert 'help' in [cmd['name'] for env in commands.values() for cmd in env]

@pytest.mark.asyncio
async def test_error_handling(server):
    """Test system error handling."""
    # Simulate API error
    server.api.openpipe.create_completion.side_effect = Exception("API Error")
    
    # Should handle error gracefully
    await server.process_exo_loop()
    
    # Verify error was logged
    # Here we could check the log file if we wanted to be thorough

@pytest.mark.asyncio
async def test_full_interaction_flow(server):
    """Test a complete interaction flow through the system."""
    # Setup mock responses
    server.api.openpipe.create_completion.return_value = {
        "choices": [{"message": {"content": "notes create \"Observing pet state\" --tags observation"}}]
    }
    
    # Run exo loop
    await server.process_exo_loop()
    
    # Verify the flow
    assert server.api.openpipe.create_completion.called
    state = await server.state_bridge.get_current_state()
    assert state['last_updated'] is not None

if __name__ == '__main__':
    pytest.main(['-v', __file__])