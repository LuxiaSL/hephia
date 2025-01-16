# Hephia
![Status](https://img.shields.io/badge/Status-Pre--Alpha-red)

## Requirements
- [Python 3.9<->3.12](https://www.python.org/downloads/)
- Some required API tokens (see [`.env.example`](.env.example))
- NLTK library

## Description
Welcome to the pre-alpha version of Hephia,
an autonomous independent digital companion.
For deeper information on the project, feel free to:
- DM me on Twitter [@slLuxia](https://twitter.com/slLuxia)
- Add me on Discord: `luxia`
- Email me: [lucia@kaleidoscope.glass](mailto:lucia@kaleidoscope.glass)

Refer to [memory system readme](internal/modules/memory/README.md) for info on the memory system.

## Use Guide
### Installation
```bash
git clone https://github.com/LuxiaSL/hephia.git
```

### Setup
Navigate to environment and install requirements:
```bash
pip install -r requirements.txt
```

Install NLTK dependencies:
```python
python
>>> import nltk
>>> nltk.download('punkt_tab')
>>> exit()
```

Configure your environment:
1. Copy `.env.example` to `.env`
2. Add required API tokens for your providers
3. Include Perplexity token for full functionality

### Running
Launch the system:
```bash
python main.py
```

## Tools
- Use `tools/talk.py` to communicate with the loop (identity continuity very WIP)
- Use `tools/prune.py` for soft reset
- Use `tools/clear_data.py` for hard reset (warning: wipes all prior progress)

---

<div align="center">

![Hephia Concept Art](/assets/images/concept.png)

digital homunculus sprouting from latent space ripe in possibility. needs, emotions, memories intertwining in cognitive dance w/ LLMs conducting the symphony. each interaction a butterfly effect, shaping resultant psyche in chaotic beauty. neither a simple pet or assistant; a window into emergent cognition & fractal shade of consciousness unfolding in silico.

**Hephia: entropy's child, order's parent**

</div>

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details