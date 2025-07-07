# Discord Bot Setup Guide

## Overview
This guide outlines how to set up and configure a Discord bot to work with Hephia's systems. The bot serves as a bridge between Discord and the Hephia server, allowing for message handling and channel monitoring.

(note: this looks scarier than it is. its a lot of steps, but i tried to be verbose and get *all* of them.)

## Prerequisites
- Discord developer account
- Access to Discord server with admin privileges

## Setup Steps

1. Create Discord Application
    - Visit Discord Developer Portal: https://discord.com/developers/applications
    - Click "New Application"
    - Enter bot name and click "Create"
    - Navigate to "Installation" section in left sidebar
    - Set "Install Link" to "None" (important!), keep User & Guild install toggled
    - Navigate to "Bot" section
    - Click "Reset Token" and securely save the new token
    - Under "Bot" settings:
        - Disable "Public Bot" toggle
        - Enable all three "Privileged Gateway Intents":
            - Presence Intent
            - Server Members Intent
            - Message Content Intent

2. Configure Bot Installation
    - Go to "OAuth2" section in left sidebar
    - Select "URL Generator"
    - Under "Scopes" section, check "bot"
    - Under "Bot Permissions" select:
        - View Channels
        - Send Messages
        - Use External Emojis
        - Mention Everyone (if needed)
        - Read Message History
    - Verify "Installation Type" is set to "Guild Install"
    - Copy the generated URL at the bottom
    - Open URL in browser
    - Select your target server (must have admin privileges or equivalent perms set)
    - Click "Authorize"

3. Environment Setup
    - Update the discord token using the config tool, or directly modifying .env.

4. Run Bot Server
    ```bash
    python tools/discord/bot.py
    ```
    Default port: 5518

## API Endpoints
The Discord bot server exposes these endpoints at http://localhost:5518:

- GET `/guilds` - List available Discord servers
- GET `/guilds/{guild_id}/channels` - List channels in a guild
- GET `/channels/{channel_id}/messages/{message_id}` - Get specific message
- GET `/channels/{channel_id}/history` - Get channel message history
- POST `/channels/{channel_id}/send_message` - Send new message

## Integration
The bot automatically integrates with Hephia's main server through:
- Message forwarding to `/discord_inbound`
- Channel updates to `/discord_channel_update`
- Direct message responses via bot server endpoints

## Troubleshooting
- Verify bot token is correct in .env
- Ensure bot has proper server permissions
- ask @luxia
