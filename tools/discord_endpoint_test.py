#!/usr/bin/env python3
import asyncio
import aiohttp
import json

# Configuration
BOT_HTTP_URL = "http://localhost:5518"
TIMEOUT = 10  # seconds

def log(message: str):
    print(f"[ASYNC TEST] {message}")

# --- Helper functions for each endpoint ---

async def async_get(session: aiohttp.ClientSession, url: str):
    log(f"GET {url}")
    try:
        async with session.get(url, timeout=TIMEOUT) as response:
            response.raise_for_status()
            # Read raw text to debug incomplete payloads
            raw = await response.text()
            log(f"Raw response length: {len(raw)}")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                log(f"Failed to parse JSON: {e}")
                data = None
            return data
    except Exception as e:
        log(f"GET request failed for {url}: {e}")
        return None

async def async_post(session: aiohttp.ClientSession, url: str, json_payload):
    log(f"POST {url} with payload: {json_payload}")
    try:
        async with session.post(url, json=json_payload, timeout=TIMEOUT) as response:
            response.raise_for_status()
            raw = await response.text()
            log(f"Raw POST response length: {len(raw)}")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                log(f"POST: Response returned 200 but no parseable JSON: {e}")
                data = None
            return data, response.status
    except Exception as e:
        log(f"POST request failed for {url}: {e}")
        return None, None

async def test_list_guilds(session: aiohttp.ClientSession):
    url = f"{BOT_HTTP_URL}/guilds"
    log("Testing list_guilds")
    data = await async_get(session, url)
    log(f"Guilds: {json.dumps(data, indent=2)}" if data else "No data returned for guilds.")
    return data

async def test_list_channels(session: aiohttp.ClientSession, guild_id: str):
    url = f"{BOT_HTTP_URL}/guilds/{guild_id}/channels"
    log(f"Testing list_channels for guild {guild_id}")
    data = await async_get(session, url)
    log(f"Channels for guild {guild_id}: {json.dumps(data, indent=2)}" if data else "No data returned for channels.")
    return data

async def test_get_history(session: aiohttp.ClientSession, channel_id: str, limit: int = 5):
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/history?limit={limit}"
    log(f"Testing get_history for channel {channel_id} with limit {limit}")
    data = await async_get(session, url)
    log(f"History for channel {channel_id}: {json.dumps(data, indent=2)}" if data else "No data returned for history.")
    return data

async def test_get_message(session: aiohttp.ClientSession, channel_id: str, message_id: str):
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/messages/{message_id}"
    log(f"Testing get_message for channel {channel_id}, message {message_id}")
    data = await async_get(session, url)
    log(f"Message: {json.dumps(data, indent=2)}" if data else "No data returned for message.")
    return data

async def test_send_message_immediate(session: aiohttp.ClientSession, channel_id: str, content: str):
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/send_message"
    payload = {"content": content}
    log(f"Testing send_message_immediate for channel {channel_id} with content: {content}")
    data, status = await async_post(session, url, payload)
    log(f"Send message response: status {status}, data: {json.dumps(data, indent=2)}" if data else "No data returned for send_message_immediate.")
    return data, status

# --- Edge-case testing functions ---

async def test_invalid_guild(session: aiohttp.ClientSession):
    invalid_guild_id = "000000000000000000"
    log(f"Testing invalid guild id: {invalid_guild_id}")
    data = await test_list_channels(session, invalid_guild_id)
    if data is None:
        log("Invalid guild test passed (no data returned as expected).")
    else:
        log("Unexpected data returned for invalid guild.")

async def test_invalid_channel(session: aiohttp.ClientSession):
    invalid_channel_id = "000000000000000000"
    log(f"Testing invalid channel id for history: {invalid_channel_id}")
    data = await test_get_history(session, invalid_channel_id)
    if data is None:
        log("Invalid channel test passed (no data returned as expected).")
    else:
        log("Unexpected data returned for invalid channel.")

async def test_empty_message_content(session: aiohttp.ClientSession, channel_id: str):
    log("Testing sending a message with empty content")
    data, status = await test_send_message_immediate(session, channel_id, "")
    if status != 200 or data is None:
        log("Empty message content test passed (error as expected).")
    else:
        log("Unexpected success for empty message content.")

async def test_malformed_payload(session: aiohttp.ClientSession, channel_id: str):
    # Simulate sending an invalid JSON payload by manually setting a wrong header and data.
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/send_message"
    headers = {"Content-Type": "application/json"}
    payload = "This is not a JSON object"
    log(f"Testing malformed payload to {url}")
    try:
        async with session.post(url, data=payload, headers=headers, timeout=TIMEOUT) as response:
            log(f"Malformed payload response status: {response.status}")
            raw = await response.text()
            try:
                data = json.loads(raw)
                log(f"Malformed payload response: {data}")
            except Exception:
                log("No valid JSON response received for malformed payload.")
    except Exception as e:
        log(f"Malformed payload test failed as expected: {e}")

async def test_concurrent_send_message(session: aiohttp.ClientSession, channel_id: str, content: str, count: int = 10):
    log(f"Testing concurrent send_message_immediate with {count} parallel requests.")
    tasks = [
        test_send_message_immediate(session, channel_id, f"{content} #{i+1}")
        for i in range(count)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    log("Concurrent send message results:")
    for res in results:
        if isinstance(res, Exception):
            log(f"Exception in concurrent request: {res}")
        else:
            data, status = res
            log(f"Status: {status}, Response: {json.dumps(data, indent=2) if data else None}")

async def test_fire_and_forget_send(session: aiohttp.ClientSession, channel_id: str, content: str):
    # Fire-and-forget style: we do not need to wait on any internal queue.
    url = f"{BOT_HTTP_URL}/channels/{channel_id}/send_message"
    payload = {"content": content}
    log(f"Testing fire-and-forget send to {url} with payload: {payload}")
    try:
        async with session.post(url, json=payload, timeout=TIMEOUT) as response:
            log(f"Fire-and-forget response status: {response.status}")
            raw = await response.text()
            try:
                data = json.loads(raw)
                log(f"Fire-and-forget response data: {json.dumps(data, indent=2)}")
            except Exception:
                log("No JSON response for fire-and-forget send.")
    except Exception as e:
        log(f"Fire-and-forget send failed: {e}")

# --- Main test runner ---

async def run_all_tests():
    async with aiohttp.ClientSession() as session:
        # Basic tests: list guilds and then list channels.
        guilds = await test_list_guilds(session)
        if not guilds or len(guilds) == 0:
            log("No guilds available; cannot proceed with channel-based tests.")
            return

        # Use the first guild.
        guild_id = guilds[0]["id"]
        channels = await test_list_channels(session, guild_id)
        if not channels or len(channels) == 0:
            log("No channels available in the guild; cannot proceed with further tests.")
            return

        # Select the first text channel. (Assuming channels have a key "type" equal to "text")
        text_channel = None
        for ch in channels:
            if ch.get("type") == "text":
                text_channel = ch
                break

        if text_channel is None:
            log("No text channels available in the guild; cannot proceed.")
            return

        channel_id = text_channel["id"]
        log(f"Using text channel {text_channel.get('name', 'Unnamed')} (ID: {channel_id}) for further tests.")

        # Get channel history to verify read endpoints.
        await test_get_history(session, channel_id, limit=5)

        # Test sending messages immediately.
        await test_send_message_immediate(session, channel_id, "Normal test message.")
        await test_empty_message_content(session, channel_id)
        await test_malformed_payload(session, channel_id)

        # Test concurrent send message to simulate load.
        await test_concurrent_send_message(session, channel_id, "Concurrent test message")

        # Test the fire-and-forget style send.
        await test_fire_and_forget_send(session, channel_id, "Fire-and-forget test message.")

        # Test invalid endpoints.
        await test_invalid_guild(session)
        await test_invalid_channel(session)

if __name__ == "__main__":
    log("Starting asynchronous extended test client for Discord bot endpoints.")
    asyncio.run(run_all_tests())
    log("Completed all tests.")
