# Hephia Tools

This directory contains assorted utility scripts for managing Hephia's cognitive processing.

## Available Tools

- **prune.py**: Recovery utility for cognitive processing when updates stall. Run this script to soft reset by a turn.

- **talk.py**: Interactive communication tool for engaging with Hephia's inner loop. Use this to establish direct conversation.

- **clear_data.py**: Entirely wipe each database. Optionally, use --include-logs to also wipe logs if desired.

- **discord_bot.py**: need to run this if you'd like to connect to discord; refer to [Discord Setup](discord_bot.md) for more info. 

- **actions_sdk.py**: this piece allows you to take care of various needs, and also send one-time messages alongside it to the cognitive system. use --help to view usage.

## Usage

Simply run the desired Python script from another command line while the server is up:

```bash
python tools\\prune.py
python tools\\talk.py 
python tools\\clear_data.py (optionally --include-tags) 
python tools\\discord_bot.py
python tools\\actions_sdk.py --help
```