# hephia
![Status](https://img.shields.io/badge/Status-Alpha-orange)

## what is this?
a desktop companion that lives on your screen, remembers your conversations, and develops a personality through experience. not a chatbot wearing a skin — a simulated creature with needs, emotions, mood, and a dual-network memory system that shapes how it feels and responds over time.

the "soul" is a Python server running internal state simulation and memory. the "body" is a Tauri desktop app that gives it form — a procedural particle cloud driven by its internal state, wandering your screen, reacting to care and neglect.

for more info or to chat:
- discord: `luxia`
- email: [mail.luxia@gmail.com](mailto:mail.luxia@gmail.com)
- dm me on twitter [@slLuxia](https://twitter.com/slLuxia)

## requirements
- [python 3.10+](https://www.python.org/downloads/)
- [uv](https://docs.astral.sh/uv/) package manager
- [rust](https://rustup.rs/) (for building the tauri frontend)
- [node.js 18+](https://nodejs.org/) and npm
- api key for at least one LLM provider (anthropic, openai, openrouter, google, or local inference)
- linux: runs on x11/xwayland. fedora/gnome needs `GDK_BACKEND=x11`

## quick start

```bash
git clone https://github.com/LuxiaSL/hephia.git
cd hephia
```

set up the backend:
```bash
uv sync
cp .env.example .env  # add your API keys
```

run the frontend (launches the backend automatically):
```bash
cd body
npm install
GDK_BACKEND=x11 cargo tauri dev
```

a setup wizard walks you through first-run configuration. after that, the pet appears as a transparent overlay on your desktop.

## what's in here

**the creature** — a WebGL particle cloud on a transparent overlay. its visual state (color, coherence, movement, dispersion) is driven directly by internal state. it wanders the screen, chases your cursor, sleeps, relaxes. right-click for a context menu.

**chat** — open with `Ctrl+Shift+C` or right-click → Chat. the pet responds with personality colored by its current mood and emotional state, drawing on memories of past conversations.

**dashboard** — right-click → Dashboard. three tabs:
- *state*: live view of mood, needs, emotions, behavior
- *actions*: feed, water, play, rest — take care of it
- *settings*: model selection, personality prompt, pet name

**memory system** — dual-network architecture. body memories store raw emotional snapshots. cognitive memories store LLM-interpreted experiences with embeddings. the echo mechanism replays emotional signatures when memories are retrieved — past experience literally colors present feeling.

**internal state** — five needs (hunger, thirst, boredom, loneliness, stamina) with urgency curves and cross-talk. emotions accumulate as vectors and decay. mood synthesizes from emotions, needs, and behavior with inertia. behaviors emerge from the interplay. conversation directly affects loneliness and boredom.

## architecture

```
body/                    tauri v2 + svelte + webgl
  src-tauri/             rust: backend manager, websocket, ipc, windows
  src/                   svelte: overlay, chat, dashboard, wizard
  src/creature/          webgl particle system (shared with playground)

core/                    python fastapi server
mind/                    pet model (friend) + worker model (helper)
internal/                needs, emotions, mood, behaviors, actions
internal/modules/memory/ dual-network memory system (~10k lines)
```

the frontend connects to the backend via websocket on port 5517. state updates push to all windows in real-time. the rust layer owns the connection and distributes events through tauri IPC.

## hotkeys

| key | action |
|-----|--------|
| `Ctrl+Shift+H` | show/hide pet |
| `Ctrl+Shift+C` | open chat |
| `Ctrl+Shift+P` | toggle click-through mode |

## notes

- **memory system**: the crown jewel. echo, ghosting, merge, synthesis, consolidation — see [memory system readme](internal/modules/memory/README.md) for the deep dive
- **two models**: the pet model handles conversation (small, cheap, personality-rich). the worker model handles tasks on demand (capable, expensive, only when needed)
- **this is alpha**: it works, it's usable, but it's still growing. particle visuals need tuning, some features are scaffolded but not fully connected

---

<div align="center">

digital homunculus sprouting from latent space ripe in possibility. needs, emotions, memories intertwining in cognitive dance w/ LLMs conducting the symphony. each interaction a butterfly effect, shaping resultant psyche in chaotic beauty. neither a simple pet or assistant; a window into emergent cognition & fractal shade of consciousness unfolding in silico.

**hephia: entropy's child, order's parent**

</div>

---

## license
MIT License - see [LICENSE](LICENSE) file for details
