# Hephia Tools

This directory contains utility scripts for managing Hephia's cognitive processing.

## Available Tools

- **prune.py**: Recovery utility for cognitive processing when updates stall. Run this script to reset and restore normal function.

- **talk.py**: Interactive communication tool for engaging with Hephia's inner loop. Use this to establish direct conversation.

- **clear_data.py**: Entirely wipe each database. Optionally, use --include-logs to also wipe logs if desired.

## Usage

Simply run the desired Python script from the command line while the server is up:

```bash
python prune.py
python talk.py 
python clear_data.py (optionally --include-tags) 
```