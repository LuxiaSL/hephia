# Hephia
![Status](https://img.shields.io/badge/Status-Pre--Alpha-red)

## Description
Welcome to the pre-alpha version of Hephia,
an autonomous independent digital companion.
For deeper information on the project, feel free to:
- DM me on Twitter [@slLuxia](https://twitter.com/slLuxia)
- Add me on Discord: `luxia`
- Email me: [lucia@kaleidoscope.glass](mailto:lucia@kaleidoscope.glass)

Refer to [memory system readme](internal/modules/memory/README.md) for info on the memory system.

## Requirements
- [Python 3.9-3.12](https://www.python.org/downloads/)
- Some required API tokens (see [`.env.example`](.env.example))

## Installation & Setup

### Quick Start (Recommended)
Run our setup script which handles everything automatically:

```bash
# Clone the repository
git clone https://github.com/LuxiaSL/hephia.git
cd hephia

# Run the setup script
python install.py
```

The script will:
1. Install uv package manager
2. Create a virtual environment
3. Install all dependencies 
4. Download required NLTK data
5. Create a .env file from the template

After setup completes, activate your environment:
```bash
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Manual Setup (Advanced)
If you prefer to set things up manually:

```bash
# Clone the repository
git clone https://github.com/LuxiaSL/hephia.git
cd hephia

# Install uv package manager
pip install uv

# Create a virtual environment
uv venv .venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install .

# Install NLTK data
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# Create .env file
cp .env.example .env
```

Edit the `.env` file to add your API tokens.

### Running
Launch the system:
```bash
python main.py
```
#### Note: The terminal tends to be buggy on Unix systems due to consistent drawing; if something fudges, just resize the window and it'll settle.

## Tools
### [Tools README](tools/README.md)
- Use `tools\actions_sdk.py` to perform different interactions and take care of needs
- Use `tools\talk.py` to communicate with the loop
- Use `tools\prune.py` for soft reset
- Use `tools\clear_data.py` for hard reset (warning: wipes all prior progress)
---

<div align="center">

![Hephia Concept Art](/assets/images/concept.png)

digital homunculus sprouting from latent space ripe in possibility. needs, emotions, memories intertwining in cognitive dance w/ LLMs conducting the symphony. each interaction a butterfly effect, shaping resultant psyche in chaotic beauty. neither a simple pet or assistant; a window into emergent cognition & fractal shade of consciousness unfolding in silico.

**Hephia: entropy's child, order's parent**

</div>

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details