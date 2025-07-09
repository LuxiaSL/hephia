# hephia
![Status](https://img.shields.io/badge/Status-Pre--Alpha-red)

## what is this?
an autonomous digital companion that learns, remembers, and grows. runs locally with your choice of LLM providers.

for more info or to chat:
- discord: `luxia`
- email: [lucia@kaleidoscope.glass](mailto:lucia@kaleidoscope.glass)
- dm me on twitter [@slLuxia](https://twitter.com/slLuxia)

## requirements
- [python 3.9-3.12](https://www.python.org/downloads/)
- api keys for desired provider

## quick start

```bash
git clone https://github.com/LuxiaSL/hephia.git
cd hephia
python launch.py
```

that's it! the launcher handles everything:
- creates virtual environment if needed
- installs dependencies automatically  
- gives you a menu to run whatever you want

## what can you run?

**main server** - the brain. runs hephia's core agent with memory, emotions, and thoughts

**monitor** - pretty TUI to watch hephia think in real-time. shows internal loop, simulated internal state, and ongoing summary.

**config tool** - edit settings, manage API keys, add new models from providers, tweak prompts without diving into files

**discord bot** - connects hephia to discord so you can chat with it there too (go to [discord_bot.md](tools/discord/discord_bot.md) to see setup instructions)

**tools** - collection of utilities:
- maintenance: reset memory, clear data, soft reset
- interaction: send messages, trigger actions (take care of/monitor needs)
- utilities: collect logs, debug stuff

## background services (optional but nice)

install as system services:

```bash
python install.py --service
```

this sets up user services that:
- start when you log in
- run quietly in background
- can be controlled with system tools
- handle dependencies (discord bot waits for main server)

install specific services:
```bash
python install.py --service main        # just the main server
python install.py --service discord     # just discord bot  
python install.py --service main,discord # both
```

## if launcher doesn't work (manual setup)

the launcher should handle 99% of cases, but if you need manual control:

```bash
# install uv package manager
pip install uv

# create virtual environment  
uv venv .venv

# activate it
# windows:
.venv\Scripts\activate
# mac/linux:
source .venv/bin/activate

# install everything
uv pip install .

# download required language data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('maxent_ne_chunker'); nltk.download('maxent_ne_chunker_tab'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('words')"
python -m spacy download en_core_web_sm

# run main server
python main.py

# run monitor (separate terminal)  
python -m client.tui

# run config tool (separate terminal)
python -m client.config
```

## notes

- **memory system**: check [memory system readme](internal/modules/memory/README.md) for deep dive
- **multiple instances**: you can run server + monitor + discord all at once

---

<div align="center">

![Hephia Concept Art](/assets/images/concept.png)

digital homunculus sprouting from latent space ripe in possibility. needs, emotions, memories intertwining in cognitive dance w/ LLMs conducting the symphony. each interaction a butterfly effect, shaping resultant psyche in chaotic beauty. neither a simple pet or assistant; a window into emergent cognition & fractal shade of consciousness unfolding in silico.

**hephia: entropy's child, order's parent**

</div>

---

## license
MIT License - see [LICENSE](LICENSE) file for details