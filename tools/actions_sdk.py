"""
tools/actions_sdk.py

Provides both programmatic and CLI interfaces for interacting with Hephia's
action system in a stateless manner with minimal dependencies.
"""

import os
import sys
import asyncio
import click
import aiohttp
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Attempt to import yaml if available, else fallback to JSON.
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Configure basic logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("hephia-actions")


# Exception classes.
class ActionClientError(Exception):
    """Base exception for action client errors."""
    pass

class ConnectionError(ActionClientError):
    """Raised when connection to server fails."""
    pass

class ActionError(ActionClientError):
    """Raised when action execution fails."""
    pass

class ConfigurationError(ActionClientError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class ActionResult:
    """Structured result from action execution."""
    success: bool
    message: str
    state_changes: Dict[str, Any]
    memory_created: bool = False


class ActionClient:
    """
    Client for interacting with Hephia's action system.

    Features:
    - Async operations with exponential backoff retries.
    - Automatic retries with jitter.
    - Simple error handling and configuration management.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: float = 30.0, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> 'ActionClient':
        """
        Create a client instance from a configuration file or environment variables.
        The default file path is ./config/actions.json unless overridden.
        """
        if not config_path:
            # Use the directory of the current script as the base for the default config path.
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.environ.get(
                'HEPHIA_CONFIG',
                str(script_dir / 'config' / 'actions.json')
            )

        config_path = Path(config_path)
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            content = config_path.read_text(encoding='utf-8')
            # Use YAML if available and file extension indicates YAML.
            if HAS_YAML and config_path.suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(content)
            else:
                # Fallback to JSON.
                config = json.loads(content)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config: {e}")

        required = ['base_url']
        missing = [key for key in required if key not in config]
        if missing:
            raise ConfigurationError(f"Missing required config keys: {missing}")

        return cls(
            base_url=config['base_url'],
            api_key=config.get('api_key'),
            timeout=config.get('timeout', 30.0),
            max_retries=config.get('max_retries', 3)
        )

    async def _ensure_session(self):
        """Ensure an aiohttp session exists."""
        if self.session is None:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            self.session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None,
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request with retries and exponential backoff.

        Raises:
            ConnectionError if the request fails after the maximum number of retries.
        """
        await self._ensure_session()
        url = f"{self.base_url}/v1/actions/{endpoint.lstrip('/')}"
        attempt = 0
        while attempt < self.max_retries:
            try:
                async with self.session.request(method, url, json=data, params=params) as response:
                    try:
                        response_data = await response.json()
                    except Exception as e:
                        raise ActionError(f"Failed to decode JSON response: {e}")
                    if response.status >= 400:
                        error_msg = response_data.get('detail', 'Unknown error')
                        if response.status == 404:
                            raise ActionError(f"Action not found: {error_msg}")
                        elif response.status == 400:
                            raise ActionError(f"Invalid request: {error_msg}")
                        else:
                            raise ActionError(f"Request failed with status {response.status}: {error_msg}")
                    return response_data
            except (aiohttp.ClientError, ActionError) as e:
                attempt += 1
                if attempt >= self.max_retries:
                    raise ConnectionError(f"Failed to connect after {self.max_retries} attempts: {e}")
                backoff_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Request failed (attempt {attempt}/{self.max_retries}): {e}. "
                               f"Retrying in {backoff_time:.1f}s")
                await asyncio.sleep(backoff_time)

    async def get_available_actions(self) -> Dict[str, Any]:
        """Get information about available actions."""
        return await self._request('GET', '')

    async def get_action_status(self, action_name: str) -> Dict[str, Any]:
        """Get the status of a specific action."""
        return await self._request('GET', f'{action_name}/status')

    async def execute_action(self, action_name: str, message: Optional[str] = None, **parameters: Any) -> ActionResult:
        """
        Execute an action with an optional message and parameters.

        Returns:
            An ActionResult instance containing the execution results.
        """
        data = {"action": action_name, "parameters": parameters}
        if message:
            data["message"] = message

        result = await self._request('POST', action_name, data=data)
        return ActionResult(
            success=result.get('success', False),
            message=result.get('message', ''),
            state_changes=result.get('state_changes', {}),
            memory_created=result.get('memory_created', False)
        )

    async def test_connection(self) -> bool:
        """
        Test connectivity to the server by fetching the available actions.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            await self.get_available_actions()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
        
    async def get_internal_state(self) -> Dict[str, Any]:
        """Get the internal state of the action system."""
        try:
            return await self._request('GET', 'state')
        except Exception as e:
            logger.error(f"Failed to get internal state: {e}")
            return {}


# CLI Implementation using click.
@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Hephia Action System CLI."""
    if verbose:
        logger.setLevel(logging.DEBUG)
    ctx.ensure_object(dict)
    try:
        ctx.obj['client'] = ActionClient.from_config(config)
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def test_connection(ctx):
    """Test connectivity to the Hephia server."""
    async def _test():
        client: ActionClient = ctx.obj['client']
        success = await client.test_connection()
        if success:
            click.echo("Connection successful!")
        else:
            click.echo("Connection failed.")
        await client.close()
    asyncio.run(_test())


@cli.command()
@click.pass_context
def list(ctx):
    """List available actions."""
    async def _list():
        client: ActionClient = ctx.obj['client']
        try:
            actions = await client.get_available_actions()
            header = f"{'Name':20} {'Description':40} {'Status':15} {'Last Used':20}"
            click.echo(header)
            click.echo("-" * len(header))
            for name, info in actions.items():
                # Determine status info.
                status = "Ready"
                stat_info = info.get('status', {})
                if stat_info.get('on_cooldown'):
                    remaining = stat_info.get('remaining_cooldown', 0)
                    status = f"Cooldown ({remaining:.1f}s)"
                last_execution = stat_info.get('last_execution', 0)
                last_used = "Never" if not last_execution else datetime.fromtimestamp(last_execution).strftime('%Y-%m-%d %H:%M:%S')
                description = info.get('description', 'No description available')
                click.echo(f"{name:20} {description:40.40} {status:15} {last_used:20}")
        except Exception as e:
            logger.error(f"Failed to list actions: {e}")
            sys.exit(1)
        finally:
            await client.close()
    asyncio.run(_list())


@cli.command()
@click.argument('action_name')
@click.option('--message', '-m', help='Message to attach to action')
@click.option('--param', '-p', multiple=True, help='Parameters in format key=value')
@click.pass_context
def execute(ctx, action_name, message, param):
    """Execute an action."""
    parameters = {}
    for p in param:
        if '=' not in p:
            logger.error(f"Invalid parameter format (expected key=value): {p}")
            sys.exit(1)
        key, value = p.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # Leave value as a string if JSON parsing fails.
            pass
        parameters[key] = value

    async def _execute():
        client: ActionClient = ctx.obj['client']
        try:
            click.echo(f"Executing action '{action_name}'...")
            result = await client.execute_action(action_name, message, **parameters)
            if result.success:
                click.echo("Action executed successfully:")
                click.echo(f"Message: {result.message}")
                if result.state_changes:
                    click.echo("State changes:")
                    for k, v in result.state_changes.items():
                        click.echo(f"  {k}: {v}")
                if result.memory_created:
                    click.echo("Memory created from action.")
            else:
                click.echo("Action execution failed:")
                click.echo(result.message)
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to execute action: {e}")
            sys.exit(1)
        finally:
            await client.close()
    asyncio.run(_execute())


@cli.command()
@click.argument('action_name')
@click.pass_context
def status(ctx, action_name):
    """Get status of an action."""
    async def _status():
        client: ActionClient = ctx.obj['client']
        try:
            stat = await client.get_action_status(action_name)
            click.echo(f"Status for action '{action_name}':")
            status_data = stat.get('status', stat)
            for key, value in status_data.items():
                if key == 'last_execution' and value:
                    value = datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                click.echo(f"  {key}: {value}")
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            sys.exit(1)
        finally:
            await client.close()
    asyncio.run(_status())

@cli.command()
@click.pass_context
def get_state(ctx):
    """Get the internal state."""
    async def _get_state():
        client: ActionClient = ctx.obj['client']
        try:
            state = await client.get_internal_state()
            click.echo("Internal State:")

            # Muffle the emotional_state to avoid overwhelming output
            if 'emotional_state' in state and isinstance(state['emotional_state'], List) and len(state['emotional_state']) > 5:
                state['emotional_state'] = state['emotional_state'][:5]  # Cap to top 5
                click.echo("Note: Emotional state is capped to the first 5 entries for brevity.")

            needs_satisfaction = []
            for need, details in state['needs'].items():
                if isinstance(details, dict) and 'satisfaction' in details:
                    pct = details['satisfaction'] * 100
                    needs_satisfaction.append(f"{need}: {pct:.2f}%")
            
            needs_str = ", ".join(needs_satisfaction) if needs_satisfaction else "None"

            # Format emotional state for display
            emotional_state_str = ", ".join([f"{e['name']}: i={e['intensity']:.2f}, v={e['valence']:.2f}, a={e['arousal']:.2f}"
                                                for e in state['emotional_state']])
            if not emotional_state_str:
                emotional_state_str = "None"

            state_str = f"""Mood: {state['mood']['name']} (v: {state['mood']['valence']:.2f}, a: {state['mood']['arousal']:.2f})
Needs: {needs_str}
Behavior: {state['behavior']['name']} (active: {state['behavior']['active']})
Emotional State: {emotional_state_str}
"""

            click.echo(state_str)

        except Exception as e:
            logger.error(f"Failed to get internal state: {e}")
            sys.exit(1)
        finally:
            await client.close()
    asyncio.run(_get_state())

def main():
    """CLI entry point."""
    cli(obj={})


if __name__ == '__main__':
    main()
