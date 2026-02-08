# Hephia Desktop Pet: Revised Implementation Specification

**Version:** 2.0
**Date:** February 8, 2026
**Status:** Planning
**Based on:** Direct codebase analysis of current Hephia (53,244 LOC)

---

## Executive Summary

Hephia is a desktop companion with simulated internal states and a sophisticated memory network that creates genuine personality through emergence. The existing codebase contains a complete "soul" (internal state simulation + dual-network memory system) that has never had a "body" (visual representation). This spec defines how to refactor the existing codebase into a shippable desktop pet while preserving the memory network's richness.

**Key Decisions:**
- **Refactor in-place**, not rewrite from scratch. The memory system is 13,111 lines of tuned, working code with emergent behaviors. A rewrite risks losing them.
- **Keep the full memory lifecycle** (echo, ghosting, merge, conflict synthesis, consolidation). This is what makes "lifelong" real.
- **Simplify research instrumentation** (semantic density NLP pipeline, state metrics). Cut ~50% of memory system lines while keeping all behavioral mechanisms.
- **Two-model architecture** for the mind layer: personality-rich friend model + capable worker model.
- **Separate visual frontend** (Tauri) consuming a Python soul server via WebSocket/REST.

---

## Part 1: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     BODY (Tauri Frontend)                        │
│  Desktop window, sprite animation, chat UI, system tray         │
│  Consumes Soul Server via WebSocket + REST                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket / HTTP
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SOUL SERVER (Python/FastAPI)                  │
│  Exposes state, accepts commands, coordinates all systems        │
├────────────┬─────────────┬──────────────┬───────────────────────┤
│  INTERNAL  │   MEMORY    │    MIND      │   INFRASTRUCTURE      │
│  STATE     │   SYSTEM    │    LAYER     │                       │
│            │             │              │                       │
│ Emotions   │ Body Net    │ Pet Model    │ Event Dispatcher      │
│ Mood       │ Cognitive   │  (friend)    │ API Clients           │
│ Needs      │  Net        │              │ Config                │
│ Behaviors  │ Echo        │ Worker Model │ Logging               │
│            │ Ghosting    │  (helper)    │ Persistence           │
│            │ Merge       │              │                       │
│            │ Conflict    │ Memory       │                       │
│            │ Consolidate │  Formation   │                       │
│            │ Retrieval   │              │                       │
└────────────┴─────────────┴──────────────┴───────────────────────┘
```

---

## Part 2: What to Keep, Simplify, and Drop

### 2.1 Internal State Simulation (~3,248 lines → ~2,800 lines)

**Source:** `internal/modules/`

These systems are complete, working, and mostly decoupled via events. Extract as a bundle with light cleanup.

| Component | File | Lines | Action | Notes |
|-----------|------|-------|--------|-------|
| EmotionalProcessor | `emotions/emotional_processor.py` | 879 | **Keep** | Stimulus model with vectors, decay, dampening. Remove cognitive bridge logging dependency. |
| MoodSynthesizer | `emotions/mood_synthesizer.py` | 350 | **Keep** | Weighted aggregate (emotions 50%, needs 30%, behavior 20%). Already handles memory echo events. |
| NeedsManager | `needs/needs_manager.py` | 244 | **Keep as-is** | 5 needs with configurable decay. Cleanest module. |
| Need | `needs/need.py` | 71 | **Keep as-is** | Simple dataclass. |
| BehaviorManager | `behaviors/behavior_manager.py` | 325 | **Keep** | 5 behaviors with probabilistic transitions. Add visual state hooks. |
| Behavior classes | `behaviors/*.py` | ~220 | **Keep as-is** | Idle, walk, chase, relax, sleep. |
| InternalContext | `internal_context.py` | 304 | **Keep** | State access layer. Already provides `get_api_context()` for frontend consumption. |
| Internal Coordinator | `internal.py` | 225 | **Refactor** | Simplify initialization chain. Remove cognitive bridge as hard dependency. |
| ActionManager | `actions/` | ~200 | **Keep** | Action cooldowns and execution. Expand for pet-specific actions. |
| Event Dispatcher | `event_dispatcher.py` (root) | 163 | **Keep as-is** | Simple event bus. Every system depends on it. |

**Coupling to resolve:**
- EmotionalProcessor references CognitiveBridge for logging only → replace with direct logger
- Internal coordinator's async init chain depends on memory system → inject via interface
- All other coupling is event-based (clean)

**Not extracting:**
- CognitiveBridge (`cognition/cognitive_bridge.py`, 758 lines) — too tightly coupled to memory internals. Rewrite as a thin interface (see Part 2.3).

### 2.2 Memory System (~13,111 lines → ~6,500 lines)

**Source:** `internal/modules/memory/`

The memory system is the crown jewel. We keep all behavioral mechanisms (echo, ghosting, merge, conflict, consolidation) and simplify research instrumentation (semantic NLP, state metrics, bidirectional evaluation).

#### 2.2.1 Nodes (Keep, light cleanup)

| Component | Lines | Action |
|-----------|-------|--------|
| `nodes/base_node.py` | 239 | Keep as-is |
| `nodes/cognitive_node.py` | 362 | Keep, drop research-only fields |
| `nodes/body_node.py` | 226 | Keep as-is |
| `nodes/node_utils.py` | 441 | Keep (blend_states used by merge/synthesis) |
| **Subtotal** | **1,268** | **→ ~1,150** |

#### 2.2.2 Networks (Simplify connection management)

| Component | Lines | Action |
|-----------|-------|--------|
| `networks/base_network.py` | 370 | Keep |
| `networks/body_network.py` | 322 | Keep |
| `networks/cognitive_network.py` | 330 | Keep |
| `networks/connections/base_manager.py` | 1,116 | **Simplify** — reduce connection health tracking granularity, simplify update batching |
| `networks/connections/cognitive_manager.py` | 301 | Keep |
| `networks/connections/body_manager.py` | 266 | Keep |
| `networks/connections/queue_manager.py` | 603 | **Simplify** — reduce queue complexity, keep batching |
| `networks/connections/node_lock_manager.py` | 411 | Keep (prevents race conditions) |
| **Subtotal** | **3,719** | **→ ~2,800** |

Primary simplification: `base_manager.py` has extensive connection health history tracking and statistical analysis. Reduce to last-3-values history (currently unbounded). `queue_manager.py` has priority/reason tracking per update — simplify to FIFO with batch flush.

#### 2.2.3 Operations (Keep all lifecycle mechanisms)

| Component | Lines | Action |
|-----------|-------|--------|
| `operations/echo_manager.py` | 454 | **Keep as-is** — the magic |
| `operations/ghost_manager.py` | 257 | **Keep as-is** — fade/revival/pruning |
| `operations/merge_manager.py` | 383 | **Keep** — light cleanup (~360) |
| `operations/consolidation_manager.py` | 211 | **Keep as-is** — essential for lifelong |
| `operations/synthesis/conflict.py` | 365 | **Keep** — refactor shim code (~300) |
| `operations/synthesis/manager.py` | 163 | **Keep as-is** |
| `operations/synthesis/base.py` | ~30 | Keep (ISynthesisHandler interface) |
| **Subtotal** | **~1,863** | **→ ~1,700** |

#### 2.2.4 Metrics (Simplify — drop SpaCy, drop state component)

| Component | Lines | Action |
|-----------|-------|--------|
| `metrics/orchestrator.py` | 757 | **Simplify** — drop state component, simplify config (~450) |
| `metrics/semantic.py` | 1,015 | **Simplify heavily** — drop SpaCy NLP pipeline, keep embedding sim + text relevance, replace density with simple heuristic (~350) |
| `metrics/emotional.py` | 247 | **Keep as-is** |
| `metrics/temporal.py` | 208 | **Keep as-is** |
| `metrics/strength.py` | 173 | **Keep as-is** |
| `metrics/state.py` | 337 | **Drop** — emotional metrics capture most of this signal |
| **Subtotal** | **2,737** | **→ ~1,430** |

**Semantic metrics detail — what stays vs goes:**

**Keep:**
- `embedding_similarity` — cosine similarity of embeddings (core retrieval)
- `text_relevance` with entity boost, weighted words, phrase matching (~130 lines)
- `semantic_cohesion` — pairwise sentence similarity (optional, ~80 lines)

**Replace:**
- Full NLP density pipeline (SpaCy dependency, syntactic/semantic/discourse surprise, Bayesian-optimized transforms) → simple information richness heuristic: `unique_words / total_words * entity_boost * length_factor` (~60 lines)

**Drop:**
- Cluster analysis (barely used, marked "EXPANSION POINT")
- State metrics calculator entirely

#### 2.2.5 Database & Support (Keep, minor cleanup)

| Component | Lines | Action |
|-----------|-------|--------|
| `db/schema.py` | 179 | Keep |
| `db/operations.py` | 739 | Keep |
| `db/managers.py` | 298 | Keep |
| `embedding_manager.py` | 261 | Keep (local sentence-transformers + API fallback) |
| `async_lru_cache.py` | 192 | Keep |
| `state/references.py` | ~80 | Keep |
| `state/signatures.py` | ~75 | Keep |
| **Subtotal** | **~1,824** | **→ ~1,750** |

#### 2.2.6 Orchestrator (Rewrite)

| Component | Lines | Action |
|-----------|-------|--------|
| `memory_system.py` | 1,637 | **Rewrite** as simpler coordinator (~700). Current orchestrator handles too many concerns (event listening, LLM calls for formation, significance evaluation, retrieval, reflection, meditation). Split into focused pieces. |

**New orchestrator responsibilities (only):**
- Initialize networks, operations managers, metrics
- Coordinate periodic updates (ghost cycle, consolidation cycle)
- Provide clean retrieval interface
- Delegate memory formation to the mind layer

#### 2.2.7 Memory System Summary

| Layer | Current | Target | Reduction |
|-------|---------|--------|-----------|
| Nodes | 1,268 | 1,150 | 9% |
| Networks + Connections | 3,719 | 2,800 | 25% |
| Operations (lifecycle) | 1,863 | 1,700 | 9% |
| Metrics | 2,737 | 1,430 | 48% |
| Database + Support | 1,824 | 1,750 | 4% |
| Orchestrator | 1,637 | 700 | 57% |
| **Total** | **13,048** | **~9,530** | **27%** |

*Note: Previous analysis estimated ~6,500 lines target. Revised upward after reading the connection management and lifecycle code — more of it is load-bearing than initially assessed. The 9,530 target preserves all behavioral mechanisms while cutting research instrumentation.*

### 2.3 New Cognitive Bridge (~300 lines, replaces 758)

The current CognitiveBridge has hard imports of memory system internals and mixes too many concerns (memory retrieval, meditation, significance evaluation, state tracking). Replace with a thin interface:

```python
class PetCognitiveBridge:
    """Thin bridge between mind layer and internal state + memory."""

    async def retrieve_memories(self, query: str, current_state: InternalState, limit: int = 5) -> List[RetrievedMemory]:
        """Query memory system, trigger echoes, return relevant memories."""

    async def form_memory(self, content: str, significance: float, source: str) -> Optional[str]:
        """Store a new memory if significant enough."""

    async def get_state_context(self) -> Dict[str, Any]:
        """Get current internal state formatted for LLM consumption."""

    async def introspect(self, topic: str) -> Optional[IntrospectionResult]:
        """Pet reflects on memories related to a topic, affecting mood/emotion."""
```

The `introspect()` method preserves the meditation/reflection capability from the current system — the pet can occasionally "think about" past experiences and have its mood shift as a result. This is what makes it feel alive between conversations.

### 2.4 Memory Formation Pipeline (~400 lines, ported from brain/cognition/memory/)

**Source:** `brain/cognition/memory/manager.py` (475 lines) + `brain/cognition/memory/significance.py` (441 lines)

Port the neuromorphic memory formation pattern:
1. Event triggers memory check (conversation, action, state change)
2. LLM generates memory prose from the experience
3. Significance evaluation (heuristic + metrics-based)
4. If significant → store in cognitive network with embedding
5. If not → discard (neuromorphic pruning)

**Simplifications:**
- Drop interface-specific significance thresholds (pet has one interface)
- Drop event-driven async metrics evaluation → direct call
- Simplify heuristic scoring
- Keep LLM-based content generation (this is what makes memories feel like *memories* rather than log entries)

### 2.5 Mind Layer (~800 lines, new)

Two-model architecture as described in the original spec. This is new code.

```
mind/
├── pet_model.py          # Small/local model interface (friend layer)
├── worker_model.py       # Capable model interface (task layer)
├── task_router.py        # Intent classification + handoff
├── conversation.py       # Conversation state management
└── prompts/              # Personality prompts, state templates
```

**Pet Model (Friend Layer):**
- All user-facing interaction
- Receives: current internal state, relevant memories, conversation history
- Produces: personality-rich responses colored by mood/emotion
- Model: small/local (Llama 8B, Phi, etc.) or cloud (cheap tier)
- Always available, low latency

**Worker Model (Helper Layer):**
- On-demand task execution
- Receives: task specification + minimal context
- Produces: structured results for pet to present
- Model: Claude Sonnet/Opus, GPT-4 class
- Only fires when user requests capability

**Task Router:**
- Pet model classifies user intent (chat vs task)
- If task → build spec, hand to worker, pet presents results
- If chat → direct pet response with state/memory context

### 2.6 Server Layer (~500 lines, rewrite)

**Source reference:** `core/server.py` (692 lines)

Rewrite for pet-specific needs. The current server has Discord routing, complex action pipeline, and TUI-specific data assembly. The pet server is simpler.

```
API Surface:

WebSocket:
  /ws/state              ← Stream internal state changes
  /ws/chat               ← Bidirectional chat messages

REST:
  GET  /state            ← Complete state snapshot
  POST /chat             ← Send message, get response
  POST /action/{name}    ← Trigger pet action (feed, play, pet, etc.)
  GET  /memory/recent    ← Recent memories
  GET  /memory/search    ← Search memories by query
  POST /worker/task      ← Submit task to worker model
  GET  /worker/{id}      ← Check task status
  GET  /settings         ← Current settings
  PUT  /settings         ← Update settings
```

### 2.7 Infrastructure (Extract from current)

| Component | Source | Lines | Action |
|-----------|--------|-------|--------|
| API Clients | `api_clients.py` | 1,315 | **Extract** — keep OpenAI, Anthropic, Local, OpenRouter. Drop Chapter2, OpenPipe, Perplexity, Discord-specific. (~800 lines) |
| Config | `config.py` | ~200 core | **Extract + simplify** for pet-specific settings |
| Loggers | `loggers/` | 465 | **Keep** — already clean |
| State Bridge | `core/state_bridge.py` | ~150 | **Adapt** — session persistence for pet |
| Prompt Loader | `brain/prompting/loader.py` | 202 | **Keep** — YAML template system is useful |
| Notes System | `brain/environments/notes.py` | 1,465 | **Port simplified** (~300 lines) — pet remembers user preferences, keeps simple notes. Drop command infrastructure, keep SQLite CRUD + search. |

### 2.8 What We're Dropping Entirely

| Component | Lines | Why |
|-----------|-------|-----|
| `brain/core/processor.py` | 466 | Autonomous command loop — wrong paradigm for pet |
| `brain/commands/preprocessor.py` | 869 | LLM→command parsing — pet doesn't use commands |
| `brain/commands/model.py` | 130 | Command dataclasses |
| `brain/interfaces/exo.py` | 602 | ExoProcessor — agent loop |
| `brain/interfaces/discord.py` | 446 | Discord interface |
| `brain/environments/discord.py` | 771 | Discord environment |
| `brain/environments/web.py` | 558 | Web browsing |
| `brain/environments/search.py` | 188 | Web search |
| `brain/environments/base_environment.py` | 306 | Command environment base |
| `brain/environments/environment_registry.py` | 109 | Environment routing |
| `brain/environments/terminal_formatter.py` | 333 | Terminal output formatting |
| `brain/interfaces/exo_utils/` | ~818 | HUD, conversation state for agent |
| `brain/utils/tracer.py` | 192 | Brain tracing (keep loggers instead) |
| `core/discord_service.py` | ~200 | Discord bot integration |
| `client/tui/` | 2,530 | Terminal UI (replaced by Tauri frontend) |
| `tools/` | 17,341 | Utilities, maintenance scripts |
| `internal/modules/cognition/cognitive_bridge.py` | 758 | Rewritten as thin bridge (Part 2.3) |
| `metrics/state.py` | 337 | State metrics calculator |
| SpaCy NLP pipeline in `metrics/semantic.py` | ~650 | Research-grade semantic analysis |
| **Total dropped** | **~27,600** | |

### 2.9 What We're Keeping from brain/ (selective port)

| Component | Source | Lines | Purpose in Pet |
|-----------|--------|-------|----------------|
| Memory formation | `brain/cognition/memory/manager.py` | 475 → 250 | Neuromorphic memory creation |
| Significance analysis | `brain/cognition/memory/significance.py` | 441 → 150 | Memory pruning decisions |
| Notification system | `brain/cognition/notification.py` | 94 | Cross-layer awareness (pet ↔ worker) |
| Meditation/Reflection core logic | `brain/environments/meditation.py` + `reflection.py` | 611 → 200 | Pet introspection (stripped of command infrastructure) |
| Action environment core | `brain/environments/action.py` | 299 → 200 | Pet actions (feed, play, rest, etc.) |

---

## Part 3: Revised Size Estimates

| Component | Current | Target | Notes |
|-----------|---------|--------|-------|
| Internal State | 3,248 | 2,800 | Light cleanup, decouple cognitive bridge |
| Memory System | 13,111 | 9,530 | Keep lifecycle, simplify metrics/orchestrator |
| Cognitive Bridge | 758 | 300 | Rewrite as thin interface |
| Memory Formation | 916 | 400 | Port + simplify from brain/cognition/ |
| Mind Layer | 0 | 800 | New (pet model + worker + router) |
| Server/API | 692 | 500 | Rewrite for pet |
| Infrastructure | ~2,100 | ~1,800 | API clients, config, loggers, state bridge |
| Introspection | 611 | 200 | Port meditation/reflection core |
| Notes | 1,465 | 300 | Simplified note storage |
| Event Dispatcher | 163 | 163 | Keep as-is |
| Prompt System | 202 | 202 | Keep as-is |
| **Total Python backend** | **~23,266** | **~16,995** | **~27% reduction** |

Plus: Tauri frontend (separate, not counted here).

---

## Part 4: Refactoring Approach

### 4.1 Branch, Don't Fork

Create a `pet` branch from current `main`. Work incrementally:

1. **Phase 0: Preparation** — Set up the branch, establish test harness for memory system (verify echo/ghost/merge behaviors before and after changes)
2. **Phase 1: Strip** — Remove everything in the "dropping" list. Get a clean, smaller codebase.
3. **Phase 2: Restructure** — Move remaining code into pet-oriented directory structure.
4. **Phase 3: Simplify** — Refactor memory metrics, connection management, orchestrator.
5. **Phase 4: Build** — New mind layer, new server, new cognitive bridge.
6. **Phase 5: Connect** — Wire everything together, verify memory behaviors survived.
7. **Phase 6: Frontend** — Tauri app consuming the soul server.

### 4.2 Why Not Rewrite

- 13,000 lines of memory system with tuned thresholds, decay rates, and interaction patterns
- Echo dampening (0.75 multiplicative, 180s window), ghost thresholds (0.1/0.2/0.05), consolidation triggers (30% weak ratio), merge connection threshold (0.7 with 3-stable-history) — these values were arrived at through iteration
- Emergent behaviors from the interplay of echo + ghosting + merge + consolidation are hard to replicate from spec alone
- The event-driven architecture means most systems can be modified independently
- Git history preserves the "why" behind non-obvious decisions

### 4.3 Target Directory Structure

```
hephia/
├── soul/                           # Python backend
│   ├── main.py                     # Entry point
│   ├── config.py                   # Configuration
│   ├── event_dispatcher.py         # Event bus (from root)
│   │
│   ├── internal/                   # Internal state simulation
│   │   ├── coordinator.py          # Orchestrates all internal systems
│   │   ├── context.py              # State access layer
│   │   ├── emotions/
│   │   │   ├── processor.py        # EmotionalProcessor
│   │   │   └── mood.py             # MoodSynthesizer
│   │   ├── needs/
│   │   │   ├── manager.py          # NeedsManager
│   │   │   └── need.py             # Need dataclass
│   │   ├── behaviors/
│   │   │   ├── manager.py          # BehaviorManager
│   │   │   └── behaviors.py        # Concrete behavior classes
│   │   └── actions/
│   │       └── manager.py          # ActionManager (expanded for pet)
│   │
│   ├── memory/                     # Memory system
│   │   ├── orchestrator.py         # Simplified coordinator
│   │   ├── nodes/
│   │   │   ├── base_node.py
│   │   │   ├── body_node.py
│   │   │   ├── cognitive_node.py
│   │   │   └── node_utils.py
│   │   ├── networks/
│   │   │   ├── base_network.py
│   │   │   ├── body_network.py
│   │   │   ├── cognitive_network.py
│   │   │   └── connections/
│   │   │       ├── base_manager.py
│   │   │       ├── body_manager.py
│   │   │       ├── cognitive_manager.py
│   │   │       ├── queue_manager.py
│   │   │       └── lock_manager.py
│   │   ├── operations/
│   │   │   ├── echo_manager.py
│   │   │   ├── ghost_manager.py
│   │   │   ├── merge_manager.py
│   │   │   ├── consolidation_manager.py
│   │   │   └── synthesis/
│   │   │       ├── conflict.py
│   │   │       └── manager.py
│   │   ├── metrics/
│   │   │   ├── orchestrator.py
│   │   │   ├── semantic.py         # Simplified (no SpaCy)
│   │   │   ├── emotional.py
│   │   │   ├── temporal.py
│   │   │   └── strength.py
│   │   ├── db/
│   │   │   ├── schema.py
│   │   │   ├── operations.py
│   │   │   └── managers.py
│   │   ├── embedding_manager.py
│   │   ├── async_lru_cache.py
│   │   └── state/
│   │       ├── references.py
│   │       └── signatures.py
│   │
│   ├── mind/                       # Model interfaces (NEW)
│   │   ├── pet_model.py            # Friend layer
│   │   ├── worker_model.py         # Helper layer
│   │   ├── task_router.py          # Intent classification
│   │   ├── conversation.py         # Conversation state
│   │   ├── memory_formation.py     # Neuromorphic memory creation
│   │   ├── introspection.py        # Pet self-reflection
│   │   └── prompts/                # Personality + state templates
│   │
│   ├── bridge.py                   # PetCognitiveBridge (thin)
│   │
│   ├── api/                        # Server (NEW)
│   │   ├── server.py               # FastAPI app
│   │   ├── routes/
│   │   │   ├── state.py
│   │   │   ├── chat.py
│   │   │   ├── actions.py
│   │   │   ├── memory.py
│   │   │   └── worker.py
│   │   └── websockets/
│   │       ├── state_stream.py
│   │       └── chat_stream.py
│   │
│   ├── tools/                      # Worker tools
│   │   ├── filesystem.py
│   │   ├── notes.py                # Simplified notes
│   │   └── registry.py
│   │
│   ├── clients/                    # LLM providers
│   │   └── api_clients.py          # Trimmed multi-provider client
│   │
│   └── loggers/                    # Logging (from current)
│       └── loggers.py
│
├── body/                           # Tauri frontend (SEPARATE)
│   ├── src-tauri/
│   │   ├── Cargo.toml
│   │   └── src/main.rs
│   ├── src/                        # Web frontend
│   │   ├── index.html
│   │   ├── main.ts
│   │   ├── components/
│   │   ├── animation/
│   │   └── api/
│   │       └── soul-client.ts      # WebSocket + REST client
│   └── assets/
│       └── sprites/
│
├── data/                           # Runtime data (gitignored)
│   ├── memory.db
│   ├── notes.db
│   ├── state.json
│   └── logs/
│
└── tests/
    ├── memory/                     # Memory behavior verification
    └── integration/
```

---

## Part 5: Implementation Phases

### Phase 1: Strip & Restructure (1-2 weeks)

**Goal:** Remove everything we're dropping, move remaining code to new structure, verify nothing broke.

| Task | Effort |
|------|--------|
| Create `pet` branch | Trivial |
| Remove brain/ (except ported pieces) | Medium |
| Remove client/tui/, tools/, Discord code | Small |
| Move internal state to soul/internal/ | Medium |
| Move memory system to soul/memory/ | Medium |
| Verify memory system still initializes and runs | Medium |
| Write basic smoke tests for echo/ghost/merge | Medium |

**Milestone:** Stripped codebase compiles and memory system passes smoke tests.

### Phase 2: Simplify Memory System (2-3 weeks)

**Goal:** Simplify metrics, connection management, orchestrator. Verify behaviors survived.

| Task | Effort |
|------|--------|
| Replace semantic density NLP pipeline with simple heuristic | Medium |
| Drop state metrics calculator, update orchestrator weights | Small |
| Simplify connection base_manager (bounded health history) | Medium |
| Simplify queue_manager (FIFO with batch flush) | Medium |
| Rewrite memory_system.py orchestrator | Large |
| Refactor conflict detection shim code | Small |
| Verify echo → emotional state propagation works | Medium |
| Verify ghosting → revival cycle works | Medium |
| Verify merge + conflict synthesis flow works | Medium |

**Milestone:** Memory system at target size, all lifecycle behaviors verified.

### Phase 3: Build Mind Layer + Bridge (2-3 weeks)

**Goal:** Pet can think, talk, and remember.

| Task | Effort |
|------|--------|
| Write PetCognitiveBridge | Medium |
| Port memory formation pipeline | Medium |
| Port significance analysis (simplified) | Small |
| Port introspection (meditation/reflection core) | Medium |
| Build pet model interface | Medium |
| Build worker model interface | Medium |
| Build task router | Small |
| Build conversation state management | Small |
| Write personality prompts | Medium |
| Write state-to-natural-language templates | Small |

**Milestone:** Can chat with pet via CLI, memories form, echo affects mood, pet introspects.

### Phase 4: Server + API (1-2 weeks)

**Goal:** Soul server exposes everything the frontend needs.

| Task | Effort |
|------|--------|
| FastAPI app with WebSocket state streaming | Medium |
| REST endpoints (state, chat, actions, memory, worker) | Medium |
| Port notes system (simplified) | Small |
| Port API clients (trimmed providers) | Small |
| Wire server to all soul systems | Medium |
| Config system for pet settings | Small |

**Milestone:** Can interact with pet via HTTP/WebSocket, all state surfaceable.

### Phase 5: Visual Frontend (2-4 weeks)

**Goal:** Pet lives on the desktop.

| Task | Effort |
|------|--------|
| Tauri project setup | Small |
| WebSocket client connecting to soul server | Small |
| Sprite system (Rive or frame-based) | Large |
| Behavior → animation state mapping | Medium |
| Emotion → expression blending | Medium |
| Chat interface | Medium |
| Thought bubble system | Small |
| System tray + always-on-top | Small |
| Pet actions UI (feed, play, pet, etc.) | Medium |

**Milestone:** Animated pet on desktop that reacts to state, chats, and remembers.

### Phase 6: Polish & Integration (1-2 weeks)

**Goal:** Shippable.

| Task | Effort |
|------|--------|
| Worker task progress UI | Medium |
| Settings panel | Medium |
| Startup/shutdown lifecycle | Small |
| State persistence across restarts | Small (mostly exists) |
| Packaging/installer | Medium |
| Performance optimization | Medium |

**Milestone:** Others can install and run their own Hephia.

---

## Part 6: Technical Decisions

### Decided

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Approach | Refactor, not rewrite | Preserve tuned memory behaviors |
| Memory lifecycle | Keep all mechanisms | Essential for lifelong operation |
| Semantic analysis | Drop SpaCy, simplify | Research overhead, heavy dependency |
| State metrics | Drop | Emotional metrics cover most signal |
| Frontend | Tauri (separate project) | Lightweight, native feel, web UI |
| Embedding model | all-MiniLM-L6-v2 (local) | Already implemented, fast, good enough |
| Database | SQLite (aiosqlite) | Already implemented, proven |

### To Decide During Implementation

| Decision | Options | When |
|----------|---------|------|
| Default pet model | Llama 8B, Phi-3, Qwen2, cloud small | Phase 3 |
| Animation system | Rive, Spine, frame-by-frame | Phase 5 |
| Sprite style | Pixel art, vector, skeletal | Phase 5 |
| Worker tool permissions | Gated vs open filesystem access | Phase 4 |
| Multi-provider default | Local-first vs cloud-first | Phase 4 |

---

## Part 7: Key Thresholds & Parameters (Preserve)

These values were tuned through iteration. Do not change without testing.

| Parameter | Value | System | Purpose |
|-----------|-------|--------|---------|
| Echo dampening decay | 0.75 | Echo Manager | Prevents echo reverberation |
| Echo dampening window | 180s | Echo Manager | Time before dampening resets |
| Ghost threshold | 0.1 | Ghost Manager | Strength below → ghosted |
| Revive threshold | 0.2 | Ghost Manager | Strength above → revived |
| Prune threshold | 0.05 | Ghost Manager | Strength below → deleted |
| Merge connection threshold | 0.7 | Merge Manager | Min connection weight for merge |
| Merge stability requirement | 3 readings | Merge Manager | Connection must be stable |
| Consolidation weak ratio | 0.3 | Consolidation | >30% weak → consolidate |
| Consolidation cooldown | 300s | Consolidation | Min time between cycles |
| Activity window | 3600s | Consolidation | 1-hour recency window |
| Emotion decay rate | 0.05/cycle | Emotional Processor | Vector intensity decay |
| Mood weights | 50/30/20 | Mood Synthesizer | Emotions/needs/behavior |
| Metric weights | .425/.2/.1/.05 | Orchestrator | Semantic/emotional/temporal/strength |
| Strength decay | 0.95/update | Networks | Exponential per-update decay |
| Connection limit | 50 | Connection Manager | Max connections per node |
| Batch queue size | 5000 | Queue Manager | Connection update queue |

---

## Appendix: Glossary

| Term | Definition |
|------|------------|
| **Soul** | Python backend containing internal state, memory, mind layer |
| **Body** | Tauri frontend providing visual representation |
| **Mind** | Layer interfacing with LLM models (pet + worker) |
| **Echo** | When retrieved memories influence current emotional state |
| **Ghost** | A decayed memory that's inactive but can be revived |
| **Merge** | Combining a weak memory into a stronger related one |
| **Consolidation** | Periodic network health maintenance cycle |
| **Synthesis** | Creating a new memory that resolves conflicting memories |
| **Pet Model** | Small, personality-rich model for user interaction |
| **Worker Model** | Capable model for task execution |
| **Bridge** | Thin interface connecting mind layer to internal state + memory |
| **Introspection** | Pet reflecting on past memories, affecting current state |

---

*End of specification.*
