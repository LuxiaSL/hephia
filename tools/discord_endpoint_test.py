#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import logging
import sys
import re
from urllib.parse import quote

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
BOT_HTTP_URL = "http://localhost:5518"
TIMEOUT = 10  # seconds

# Set up detailed logging for the test suite
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("discord_endpoint_test.log")
    ]
)
logger = logging.getLogger("DiscordBotTestSuite")

def log(message: str, level=logging.INFO):
    logger.log(level, f"[ASYNC TEST] {message}")

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR HTTP REQUESTS
# ------------------------------------------------------------------------------
async def async_get(session: aiohttp.ClientSession, url: str, raise_for_status=False):
    log(f"GET {url}")
    try:
        async with session.get(url, timeout=TIMEOUT) as response:
            raw = await response.text()
            status = response.status
            log(f"Response status: {status}, raw length: {len(raw)}")
            
            if raise_for_status:
                response.raise_for_status()
                
            try:
                data = json.loads(raw)
                return data, status
            except json.JSONDecodeError as e:
                log(f"Failed to parse JSON: {e}", logging.ERROR)
                log(f"Raw response: {raw[:500]}...", logging.DEBUG)
                return raw, status
    except Exception as e:
        log(f"GET request failed for {url}: {e}", logging.ERROR)
        return None, getattr(e, 'status', None)

async def async_post(session: aiohttp.ClientSession, url: str, json_payload, raise_for_status=False):
    log(f"POST {url} with payload: {json_payload}")
    try:
        async with session.post(url, json=json_payload, timeout=TIMEOUT) as response:
            raw = await response.text()
            status = response.status
            log(f"Response status: {status}, raw length: {len(raw)}")
            
            if raise_for_status:
                response.raise_for_status()
                
            try:
                data = json.loads(raw)
                return data, status
            except json.JSONDecodeError as e:
                log(f"POST: Failed to parse JSON: {e}", logging.ERROR)
                log(f"Raw response: {raw[:500]}...", logging.DEBUG)
                return raw, status
    except Exception as e:
        log(f"POST request failed for {url}: {e}", logging.ERROR)
        # Try to extract status code if available
        if hasattr(e, 'status'):
            return None, e.status
        return None, None

# ------------------------------------------------------------------------------
# NEW ENDPOINT TESTS
# ------------------------------------------------------------------------------

async def test_list_guilds_with_channels(session: aiohttp.ClientSession):
    """Test the hierarchical guild/channel listing endpoint."""
    url = f"{BOT_HTTP_URL}/guilds-with-channels"
    log("Testing list_guilds_with_channels")
    data, status = await async_get(session, url)
    if data and status == 200:
        log(f"Guilds with channels:\n{json.dumps(data, indent=2)}")
        # Verify structure
        if isinstance(data, list):
            for guild in data:
                if 'name' not in guild or 'channels' not in guild:
                    log(f"Guild missing required fields: {guild}", logging.WARNING)
    else:
        log(f"Error fetching guilds: status={status}", logging.ERROR)
    return data, status

async def test_list_guilds(session: aiohttp.ClientSession):
    """Test the original list_guilds endpoint."""
    url = f"{BOT_HTTP_URL}/guilds"
    log("Testing list_guilds")
    data, status = await async_get(session, url)
    if data and status == 200:
        log(f"Original Guilds List:\n{json.dumps(data, indent=2)}")
    else:
        log(f"Error fetching guilds: status={status}", logging.ERROR)
    return data, status

async def test_send_by_path(session: aiohttp.ClientSession, path: str, content: str):
    """Test sending a message using the guild:channel path format."""
    url = f"{BOT_HTTP_URL}/send-by-path"
    payload = {"path": path, "content": content}
    log(f"Testing send_by_path to '{path}' with content: '{content}'")
    data, status = await async_post(session, url, payload)
    if status == 200:
        log(f"Send by path response:\n{json.dumps(data, indent=2) if isinstance(data, dict) else data}")
    else:
        log(f"Error sending message by path: status={status}", logging.ERROR)
        log(f"Response data: {data}")
    return data, status

async def test_send_by_id(session: aiohttp.ClientSession, channel_id: str, content: str):
    """Test sending a message using the original channel ID method."""
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/send_message"
    payload = {"content": content}
    log(f"Testing send message to channel ID '{channel_id}' with content: '{content}'")
    data, status = await async_post(session, url, payload)
    if status == 200:
        log(f"Send by ID response:\n{json.dumps(data, indent=2) if isinstance(data, dict) else data}")
    else:
        log(f"Error sending message by ID: status={status}", logging.ERROR)
    return data, status

async def test_get_history_by_path(session: aiohttp.ClientSession, path: str, limit: int = 10):
    """Test getting message history using the guild:channel path format."""
    # URL encode the path to handle special characters properly
    encoded_path = quote(path)
    url = f"{BOT_HTTP_URL}/history-by-path?path={encoded_path}&limit={limit}"
    log(f"Testing get_history_by_path for '{path}' with limit {limit}")
    data, status = await async_get(session, url)
    if status == 200:
        log(f"History by path:\n{json.dumps(data, indent=2) if isinstance(data, dict) else data}")
    else:
        log(f"Error getting history by path: status={status}", logging.ERROR)
    return data, status

# ------------------------------------------------------------------------------
# PATH FORMAT VALIDATION TESTS
# ------------------------------------------------------------------------------

async def test_path_format_variations(session: aiohttp.ClientSession, base_guild_name: str, base_channel_name: str):
    """Test different variations of the path format to find what works."""
    log(f"Testing path format variations for guild '{base_guild_name}' and channel '{base_channel_name}'")
    
    # Create different variations of server and channel names
    variations = [
        # Original format
        f"{base_guild_name}:{base_channel_name}",
        
        # Case variations
        f"{base_guild_name.upper()}:{base_channel_name}",
        f"{base_guild_name}:{base_channel_name.upper()}",
        f"{base_guild_name.lower()}:{base_channel_name.lower()}",
        
        # Whitespace variations
        f" {base_guild_name}:{base_channel_name}",
        f"{base_guild_name} :{base_channel_name}",
        f"{base_guild_name}: {base_channel_name}",
        f"{base_guild_name}:{base_channel_name} ",
        
        # Special character handling
        f"{base_guild_name.replace('-', '')}:{base_channel_name}",  # Remove hyphens
        f"{base_guild_name.replace('-', '_')}:{base_channel_name}",  # Replace hyphens with underscores
        
        # Channel name only (for DMs potentially)
        f"{base_channel_name}"
    ]
    
    for idx, path in enumerate(variations):
        log(f"Testing variation #{idx+1}: '{path}'")
        data, status = await test_send_by_path(session, path, f"Test message for path variation #{idx+1}")
        log(f"Result for variation #{idx+1}: status={status}, success={status==200}")
    
    # Also test the raw channel name without guild prefix
    channel_only = f"{base_channel_name}"
    log(f"Testing channel-only variation: '{channel_only}'")
    data, status = await test_send_by_path(session, channel_only, "Test message for channel-only format")
    log(f"Result for channel-only: status={status}, success={status==200}")

async def verify_endpoint_registration(session: aiohttp.ClientSession):
    """Verify which endpoints are properly registered with the bot."""
    endpoints_to_check = [
        "/guilds",                   # Original endpoints
        "/guilds-with-channels",     # New endpoints
        "/send-by-path",
        "/history-by-path",
        "/enhanced-history",
        "/find-message",
        "/reply-to-message"
    ]
    
    for endpoint in endpoints_to_check:
        url = f"{BOT_HTTP_URL}{endpoint}"
        # Try both GET and POST to see if the endpoint exists at all
        # (even if it returns an error, we just want to check if it's registered)
        for method in ["GET", "POST"]:
            log(f"Checking if {method} {endpoint} exists")
            if method == "GET":
                _, status = await async_get(session, url)
            else:
                _, status = await async_post(session, url, {"test": "payload"})
            
            # 404 means endpoint doesn't exist, anything else means it does
            # (405 Method Not Allowed would mean endpoint exists but doesn't support this method)
            log(f"{method} {endpoint}: status={status}, exists={status != 404}")

# ------------------------------------------------------------------------------
# COMPREHENSIVE DEBUGGING TESTS FOR OUR ISSUE
# ------------------------------------------------------------------------------

async def debug_specific_path(session: aiohttp.ClientSession, path: str, 
                             alternate_formats: bool = True, 
                             compare_with_id: bool = True):
    """
    Comprehensive debug of a specific channel path we're having issues with.
    Tries multiple variations and compares with direct ID access.
    """
    log(f"=== COMPREHENSIVE DEBUG FOR PATH: '{path}' ===")
    
    # 1. First verify we can GET the channel info
    log("1. Checking if we can list channels and find this path")
    guilds_data, guild_status = await test_list_guilds_with_channels(session)
    
    if guild_status != 200:
        log("Cannot list guilds, stopping debug", logging.ERROR)
        return False
    
    # Try to find this path in the returned data
    found_in_listing = False
    channel_id = None
    guild_id = None
    guild_name = None
    channel_name = None
    
    # Parse our target path
    if ":" in path:
        guild_name, channel_name = path.split(":", 1)
    else:
        # Assume it's a channel name without guild
        channel_name = path
    
    # Search in guilds data
    for guild in guilds_data:
        if (not guild_name or  
            guild_name.lower() == guild.get('name', '').lower() or
            guild_name == guild.get('id', '')):
            guild_id = guild.get('id')
            for channel in guild.get('channels', []):
                if (channel_name.lower() == channel.get('name', '').lower() or
                    channel_name == channel.get('id', '')):
                    found_in_listing = True
                    channel_id = channel.get('id')
                    path_in_listing = channel.get('path')
                    log(f"Found matching channel: id={channel_id}, listed_path='{path_in_listing}'")
                    break
            if found_in_listing:
                break
    
    # 2. Test sending with the exact path from listings if found
    if found_in_listing and 'path' in channel:
        exact_path = channel['path']
        log(f"2. Testing with exact path from listing: '{exact_path}'")
        _, status = await test_send_by_path(session, exact_path, "Test with exact path from listing")
        log(f"Send with exact listed path result: status={status}, success={status==200}")
    
    # 3. Test with our original path
    log(f"3. Testing with original path: '{path}'")
    _, status = await test_send_by_path(session, path, "Test with original path")
    log(f"Send with original path result: status={status}, success={status==200}")
    
    # 4. If we have the channel ID, test direct ID-based sending
    if compare_with_id and channel_id:
        log(f"4. Testing with direct channel ID: '{channel_id}'")
        _, id_status = await test_send_by_id(session, channel_id, "Test with direct channel ID")
        log(f"Send with direct ID result: status={id_status}, success={id_status==200}")
    
    # 5. If requested, try alternate formats
    if alternate_formats:
        if ":" in path:
            guild_part, channel_part = path.split(":", 1)
            
            alternates = [
                # Case variations
                f"{guild_part.lower()}:{channel_part.lower()}",
                f"{guild_part.upper()}:{channel_part}",
                
                # Whitespace variations
                f"{guild_part}:{channel_part.strip()}",
                f"{guild_part.strip()}:{channel_part}",
                
                # Remove special characters
                re.sub(r'[^a-zA-Z0-9]', '', guild_part) + ":" + channel_part,
                
                # Just the raw parts
                f"{guild_part}",
                f"{channel_part}"
            ]
            
            log("5. Testing alternate formats")
            for idx, alt_path in enumerate(alternates):
                log(f"Alternative #{idx+1}: '{alt_path}'")
                _, alt_status = await test_send_by_path(session, alt_path, f"Test with format variation #{idx+1}")
                log(f"Alternative #{idx+1} result: status={alt_status}, success={alt_status==200}")
    
    log("=== DEBUG COMPLETE ===")
    return True

# ------------------------------------------------------------------------------
# MAIN TEST RUNNER
# ------------------------------------------------------------------------------

async def run_targeted_debug():
    """Run a focused debug session on our specific issue path."""
    async with aiohttp.ClientSession() as session:
        log("Starting targeted debugging for the 'hephia-testing:general' issue")
        
        # 1. First verify which endpoints are registered
        await verify_endpoint_registration(session)
        
        # 2. Try comprehensive debug on our problem path
        await debug_specific_path(session, "hephia-testing:general")
        
        # 3. Try some variations of the path format
        await test_path_format_variations(session, "hephia-testing", "general")
        
        log("Targeted debugging complete.")

async def run_all_tests():
    """Run the full test suite."""
    async with aiohttp.ClientSession() as session:
        # Start with our targeted debug
        await run_targeted_debug(session)
        
        # Continue with regular tests
        guild_data, status = await test_list_guilds_with_channels(session)
        if not guild_data or status != 200:
            log("No guild data available; cannot proceed with path-based tests.", logging.ERROR)
            return

        # Extract a valid path from the guild data
        valid_path = None
        for guild in guild_data:
            if "channels" in guild and guild["channels"]:
                # Use the first available text channel in the guild
                valid_path = guild["channels"][0].get("path") or f"{guild['name']}:{guild['channels'][0]['name']}"
                break

        if not valid_path:
            log("No valid path found for testing.", logging.ERROR)
            return

        log(f"Using valid path '{valid_path}' for subsequent tests.")

        # Test basic operations with paths
        await test_send_by_path(session, valid_path, "Test message via path")
        await test_get_history_by_path(session, valid_path, limit=10)
        
        log("All tests completed.")

if __name__ == "__main__":
    log("Starting asynchronous debug for Discord bot endpoints.")
    asyncio.run(run_targeted_debug())
    log("Completed all tests.")