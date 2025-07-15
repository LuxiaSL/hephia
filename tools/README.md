# Hephia Tools

This directory contains assorted utility scripts for managing Hephia's cognitive processing.

## Some Available Tools

- **prune.py**: Recovery utility for cognitive processing when updates stall. Run this script to soft reset by a turn.

- **talk.py**: Interactive communication tool for engaging with Hephia's inner loop. Use this to establish direct conversation.

- **clear_data.py**: Entirely wipe each database. Optionally, use --include-logs to also wipe logs if desired.

- **discord_bot.py**: need to run this if you'd like to connect to discord; refer to [Discord Setup](discord_bot.md) for more info. 

- **actions_sdk.py**: this piece allows you to take care of various needs, and also send one-time messages alongside it to the cognitive system. use --help to view usage.

## Usage

Simply run launch.py and choose the desired tool you'd like to use. Handles venv and other bits for you. 