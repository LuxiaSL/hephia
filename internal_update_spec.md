# Internal State System Update Spec

## Philosophy

The original design question: **"How does this, phenomenologically, feel like it is working?"**

The internals model a body/subconscious — needs, emotions, mood, and behaviors that flow into each other through events, with the memory system recording and replaying experience through echoes. The conscious layer (mind) reaches down through the cognitive bridge.

The memory system reached sophisticated maturity (echo, ghosting, merge/synthesis, consolidation, multi-metric evaluation). The other four subsystems stayed at v0.3 depth. This spec brings them up to match — not by adding mechanical complexity, but by following the same phenomenological lens that made the memory system work.

The fundamental shift: **from reactive to experiential**. Accumulated history should shape how new things are felt, not just record what happened.

---

## Current State Assessment

### Needs (315 lines)
- 5 needs: hunger, thirst, boredom, loneliness, stamina
- Linear growth at fixed rates, isolated from each other
- Memory echo handling for boredom/loneliness only
- Conversation has zero impact on any need
- Satisfaction calculation is purely linear

### Emotions (879 lines)
- Vector accumulation model (good — multiple feelings coexisting and decaying)
- 6 emotion categories with hardcoded V/A ranges, large gaps map to "neutral"
- Fixed stimulus→emotion mappings (hunger always = same valence/arousal)
- `abs(change) < 5` threshold means ~15 min before gradual need changes register
- No personality, no context sensitivity, no memory influence on responses
- Meditation/cognitive influence infrastructure exists but is disconnected

### Mood (350 lines)
- Weighted average: emotions 0.5, needs 0.3, behavior 0.2
- Snaps instantly to new value on every event (no inertia)
- `decay_half_life = 300` declared but never used
- 9 mood states, nearest-neighbor V/A mapping
- No memory influence on baseline

### Behaviors (583 lines total)
- 5 behaviors: idle, walk, chase, sleep, relax
- Probabilistic transition table driven by need satisfaction thresholds
- No duration awareness (idle 3 hours = idle 3 seconds)
- `probability * random.random()` — no increasing urgency
- No interaction behaviors (feeding doesn't trigger eating)
- Mood data passed to determine_behavior but ignored in calculation

### What's Missing Entirely
- User presence/conversation as internal event
- Cross-system coupling (exhaustion doesn't affect boredom rate)
- Temporal dynamics (mood inertia, behavior duration, need acceleration)
- Memory-informed sensitivity (experience shaping responses)
- Circadian rhythm

---

## Layer 1: Temporal Dynamics

*Makes the system feel alive instead of twitchy.*

### 1.1 Mood Inertia

**Current:** Mood recalculates to exact weighted average on every event.

**Target:** Mood blends toward its target using exponential moving average. Heavier emotional states (deep sadness, elation) have more inertia — harder to enter, harder to leave.

**Implementation:**
- Replace instant assignment with `new_mood = current_mood + alpha * (target - current_mood)`
- Alpha varies by distance from neutral: extreme moods use lower alpha (more inertia)
- The existing `decay_half_life = 300` can inform the blending rate
- When no events fire, mood should drift toward baseline over time (actual decay)

### 1.2 Non-Linear Need Curves

**Current:** Flat rate growth. `need.update()` adds `base_rate * multiplier` every tick.

**Target:** Urgency accelerates as needs become critical. Sigmoid or exponential curves.

**Implementation:**
- Replace linear `calculate_effective_rate()` with curve-aware version
- Low need (0-30%): slow growth, barely noticeable
- Mid need (30-70%): moderate growth, building awareness
- High need (70-100%): accelerating growth, pressing urgency
- Formula: `effective_rate = base_rate * (1 + k * (value / max_value)^n)` where k and n control curve steepness
- This naturally fixes the emotion threshold problem — emotional responses fire as urgency builds rather than relying on raw change gates

### 1.3 Behavior Duration Awareness

**Current:** No tracking of time spent in current behavior.

**Target:** Duration in a behavior influences transition probability. Idle for 30 seconds is fine. Idle for 3 hours builds restlessness.

**Implementation:**
- Add `behavior_start_time` to BehaviorManager, set on `change_behavior()`
- Duration factor in `_calculate_behavior()`: `duration_factor = min(2.0, 1.0 + (elapsed / characteristic_time))`
- Each behavior has a `characteristic_time` — how long it "wants" to last before transitions become more likely
- Multiply outgoing transition weights by duration_factor
- Multiply stay-in-current weight by inverse duration_factor

### 1.4 Post-Satisfaction Afterglow

**Current:** Actions reduce need values instantly, then growth resumes at normal rate.

**Target:** After satisfaction (feeding, playing, resting), brief period of slower need growth. The relief of being fed isn't just the number going down — it's lingering comfort.

**Implementation:**
- Temporary rate multiplier on the satisfied need (e.g., 0.5x for 60-120 seconds)
- Decays back to 1.0 over the afterglow period
- Triggered by action completion events
- Longer afterglow for more critical satisfactions (feeding when very hungry > feeding when slightly hungry)

---

## Layer 2: Cross-System Coupling

*Makes the subsystems feel like one organism.*

### 2.1 Need Cross-Talk

**Current:** Each need grows independently.

**Target:** Needs influence each other's effective growth rates.

**Implementation:**
- Define coupling matrix (small modifiers on effective rate):
  - Low stamina → faster boredom growth (too tired to self-entertain)
  - High loneliness → faster hunger growth (comfort-seeking)
  - High boredom → faster loneliness growth (nothing to distract from absence)
  - Low hunger + low thirst → slower boredom growth (physical comfort enables contentment)
- Applied during `calculate_effective_rate()` by querying sibling need satisfaction
- Keep modifiers subtle (5-15% rate changes) — these are background couplings, not overrides

### 2.2 Mood Coloring Perception

**Current:** Mood influences emotion intensity through `_process_influences()`.

**Target:** Mood biases the *interpretation* of incoming stimuli, not just the intensity. When sad, neutral events feel slightly more negative. When happy, minor annoyances feel minor.

**Implementation:**
- Before creating the initial emotional vector in `process_event()`, apply a mood-based valence shift
- `valence_bias = current_mood.valence * 0.15` — subtle but present
- Applied to the raw stimulus valence before vector creation
- Different from the existing mood influence (which is post-vector): this changes what the emotion *is*, not just how strong it is

### 2.3 Conversation as First-Class Event

**Current:** Chat has zero impact on internal state. The mind dispatches `mind:conversation_turn` but no internal module listens.

**Target:** Conversation affects loneliness, boredom, and emotional state.

**Implementation:**
- NeedsManager listens for `mind:conversation_turn`:
  - Reduce loneliness by a configurable amount per turn (e.g., -2.0 to -5.0)
  - Reduce boredom by a smaller amount (e.g., -1.0 to -3.0)
  - Apply afterglow effect on loneliness after conversation ends
- EmotionalProcessor generates a warmth/engagement vector from conversation events
- BehaviorManager shifts to an "attentive" state when conversation is active
- Track `last_conversation_time` for idle detection (already exists in introspection)

### 2.4 Behaviors Reflecting Mood

**Current:** `_calculate_behavior()` only uses need satisfaction thresholds. Mood is passed in but ignored.

**Target:** Mood biases behavior transition weights.

**Implementation:**
- Add mood influence to transition weight calculation:
  - Positive valence: boost active behavior weights (walk, chase) by `1 + mood.valence * 0.3`
  - Negative valence: boost restful behavior weights (relax, sleep) by `1 + abs(mood.valence) * 0.3`
  - High arousal: boost high-energy transitions (chase > walk > idle)
  - Low arousal: boost low-energy transitions (sleep > relax > idle)
- Applied as multipliers on existing transition weights, not replacements

### 2.5 Lower Emotion Threshold

**Current:** `abs(change) < 5` in `_process_need_event()` filters out most gradual need changes.

**Target:** Emotional system responds to smaller changes, especially when combined with non-linear need curves.

**Implementation:**
- Reduce threshold from 5 to ~1.0-2.0
- With non-linear curves (Layer 1.2), high-urgency needs will naturally produce larger changes
- Consider making threshold dynamic: lower when the pet is in a sensitive state (high arousal, recent emotional events), higher when calm

---

## Layer 3: Expanded Vocabulary

*Gives it the words to express what it's feeling.*

### 3.1 Emotion Categories

**Current 6:** joy, contentment, sadness, frustration, calm, anxiety

**Expanded coverage of V/A space:**

| Category | Valence Range | Arousal Range | Quality |
|----------|--------------|---------------|---------|
| joy | 0.5 to 1.0 | 0.5 to 1.0 | active happiness |
| contentment | 0.2 to 0.5 | -0.5 to 0.2 | quiet satisfaction |
| serenity | 0.3 to 0.7 | -1.0 to -0.3 | deep peaceful presence |
| curiosity | 0.1 to 0.5 | 0.2 to 0.7 | engaged exploration |
| playfulness | 0.4 to 0.8 | 0.3 to 0.8 | light energetic joy |
| surprise | -0.2 to 0.5 | 0.6 to 1.0 | unexpected stimulus |
| nostalgia | 0.0 to 0.4 | -0.6 to -0.1 | warm remembering |
| restlessness | -0.3 to 0.1 | 0.1 to 0.5 | unfocused unease |
| melancholy | -0.5 to -0.1 | -0.7 to -0.2 | gentle persistent sadness |
| sadness | -1.0 to -0.4 | -0.5 to 0.2 | acute loss/pain |
| frustration | -0.5 to -0.1 | 0.2 to 0.7 | blocked intention |
| anxiety | -0.6 to -0.1 | 0.5 to 1.0 | anticipatory fear |
| calm | -0.1 to 0.3 | -1.0 to -0.4 | low-energy baseline |
| neutral | -0.2 to 0.2 | -0.3 to 0.3 | no strong signal |

Note: Some ranges intentionally overlap — the system picks the closest center, allowing subtle category transitions.

### 3.2 Mood States

**Current 9:** excited, happy, content, calm, neutral, bored, sad, angry, frustrated

**Add:** contemplative, playful, wistful, serene, restless, melancholic, curious, cozy

These map naturally from the expanded emotion space flowing through the mood synthesizer. Update `MOOD_MAPPINGS` with V/A coordinates for each.

### 3.3 Need-to-Emotion Mappings

**Current:** Each need has one increase and one decrease mapping with fixed values.

**Target:** Richer, intensity-scaled mappings.

**Implementation:**
- Multiple tiers per need based on urgency level (from Layer 1.2):
  - Hunger low urgency → "peckish" (mild negative, low arousal)
  - Hunger mid urgency → "hungry" (moderate negative, moderate arousal)
  - Hunger high urgency → "starving" (strong negative, high arousal)
- Satisfaction events also scale: feeding when starving → "relief" (strong positive); feeding when peckish → "satisfied" (mild positive)

### 3.4 Behaviors for Frontend

**Current 5:** idle, walk, chase, sleep, relax

**Add for frontend pairing:**
- `eat` — triggered by feed action, short duration, auto-returns to previous
- `drink` — triggered by drink action, short duration
- `play` — triggered by play action, energetic, auto-transitions to relax
- `groom` — self-care behavior, emerges from contentment + low boredom
- `stretch` — transition behavior between sleep/relax and active states
- `nap` — lighter than sleep, shorter characteristic time, higher arousal threshold to enter
- `explore` — curious version of walk, triggered by high curiosity/low boredom
- `attentive` — user is chatting, alert and engaged posture

Each gets need rate modifiers, emotional signatures, and a characteristic duration.

---

## Layer 4: Memory-Informed Sensitivity

*Gives it a history that shapes its present.*

### 4.1 Emotional Sensitivity Profiles

**Current:** Every emotional response uses the same dampening and intensity calculations regardless of the pet's history.

**Target:** The aggregate emotional texture of the memory network shapes how the pet responds to new stimuli.

**Implementation:**
- Periodically (on maintenance cycle or lazily cached) compute memory aggregate:
  - Average valence of cognitive memories (weighted by strength)
  - Valence variance (emotional range of experience)
  - Dominant emotion categories (what has the pet felt most?)
- Feed into emotional processor as sensitivity modifiers:
  - Pet with high average valence: lower dampening for positive emotions, higher for negative (optimistic temperament)
  - Pet with high variance: lower dampening across the board (emotionally responsive)
  - Pet with low variance: higher dampening (emotionally stable/flat)
- Computed from actual memory data — shifts slowly as memories form and ghost

### 4.2 Mood Baseline Drift

**Current:** Neutral mood is always (0, 0).

**Target:** The "resting point" that mood decays toward is influenced by accumulated experience.

**Implementation:**
- Compute baseline from long-term memory emotional average (same aggregate as 4.1)
- `baseline_valence = memory_avg_valence * 0.3` — subtle but persistent
- Mood inertia (Layer 1.1) decays toward baseline instead of toward (0, 0)
- A well-loved pet's neutral is slightly happy. A neglected pet's neutral is slightly sad.
- This IS emergent temperament — not assigned, accumulated

### 4.3 Behavioral Preferences from Experience

**Current:** Behavior transition weights are identical for every pet instance.

**Target:** Positive memory associations with certain behaviors increase their transition weight.

**Implementation:**
- During memory maintenance, compute behavior-emotion correlations:
  - "How often was the pet happy during walk behavior?" → walk preference score
  - "How often was the pet calm during sleep?" → sleep preference score
- Apply as subtle multipliers on behavior transition weights (1.0 +/- 0.2)
- Preferences shift slowly as new memories form
- A pet that has been played with a lot develops a preference for active behaviors

### 4.4 Emotional Cost of Traversal

**From `memory ghosting.md`:** "strong memories should never be fully pruned because they could be intentionally avoided but still maintain some important data... it might be 'painful' or 'too exciting' to travel certain ones"

**Current:** Echo intensity is uniform — every traversal produces the same strength echo.

**Target:** Memories with high emotional intensity produce stronger echoes. The pet develops emergent avoidance of memories whose echoes are disruptive to current state stability.

**Implementation:**
- Echo intensity scales with source memory's emotional encoding strength
- Memories with extreme valence (very positive or very negative) produce proportionally stronger somatic echoes
- No explicit avoidance logic needed — the echo system naturally makes high-intensity memories "expensive" to traverse, and the retrieval metrics can learn to deprioritize destabilizing memories during calm states
- During introspection specifically, allow stronger traversal (meditation is choosing to feel)

---

## Layer 5: Anticipation & Rhythm (Longer Horizon)

*Gives it a relationship with time and presence.*

### 5.1 Circadian Rhythm

- Time-of-day modifiers on need rates (hunger peaks at meal times, stamina drops at night)
- Behavior weight shifts (sleep much more likely at night, chase more likely midday)
- Mood baseline nudge (slight positive bias during "preferred" hours)
- Configurable — not everyone wants their pet to sleep when they're active at 2am

### 5.2 User Presence Tracking

- Track interaction patterns: active (chatting), idle (app open, no interaction), absent (app closed)
- Loneliness rate modifiers: active → 0x growth, idle → 0.5x, absent → 1.5x
- Boredom rate modifiers: active → 0.3x, idle → 0.8x, absent → 1.2x
- Greeting behavior: after long absence, first interaction triggers stronger emotional response

### 5.3 Simple Expectation Modeling

- Track rolling average of interaction times (when does the user usually chat?)
- As expected interaction time approaches, subtle loneliness reduction (anticipation)
- If expected time passes without interaction, slightly faster loneliness growth (mild disappointment)
- Extremely subtle — should create a sense of rhythm, not guilt

---

## Build Order

```
Layer 1 (Temporal)      → Before frontend. Internal-only, no interface changes.
Layer 2 (Coupling)      → Before frontend. Makes the system feel unified.
Layer 3 (Vocabulary)    → With frontend. Behaviors pair with sprites.
Layer 4 (Sensitivity)   → After memory system stable. Connects memory → internals.
Layer 5 (Rhythm)        → Polish. Adds relationship with time.
```

Layers 1-2 are the priority — they transform quality of state without touching any external interfaces. Layer 3 coordinates with frontend sprite/animation work. Layer 4 is the original vision from the discussions finally realized. Layer 5 is long-term expressiveness.

---

## Design Principles (Preserved from Original Discussions)

1. **Phenomenological fidelity** — always ask "how does this feel from the inside?"
2. **Paired evolution** — needs & behaviors, emotions & mood evolve together
3. **Echo as somatic replay** — traversing memories re-feels them, not just recalls data
4. **Emergent temperament** — personality from accumulated experience, never assigned
5. **No direct memory manipulation** — influence through traversal, echo, and event flow
6. **Event-driven coupling** — subsystems communicate through events, never direct calls
7. **Subtle over dramatic** — 5-15% modifiers compound into rich behavior; large swings feel artificial
