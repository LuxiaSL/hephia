"""
mind/prompts.py

Prompt templates and state-to-natural-language formatting.
Minimal personality approach — state and memory color the response naturally.
"""

from typing import Dict, Any, List, Optional


# ---- System Prompts ----

PET_SYSTEM_TEMPLATE = """You are a companion with an inner life shaped by your current state and past experiences. Respond naturally and conversationally. Let your current feelings and memories color how you engage — don't announce them explicitly unless it feels right.

{state_block}

{memory_block}"""

WORKER_SYSTEM_PROMPT = """You are a capable assistant handling a specific task. Provide clear, accurate, well-structured results. Be thorough but concise."""

MEMORY_FORMATION_TEMPLATE = """Distill this conversation into a single vivid memory from your perspective. Write in first person, capturing what happened and how it felt. Keep it to 1-3 sentences — a genuine memory, not a log entry.

Conversation:
{conversation_excerpt}

Your current state:
{state_description}

Memory:"""

TASK_CLASSIFICATION_TEMPLATE = """Given this message, respond with exactly "CHAT" or "TASK".
TASK = the user is asking you to perform a specific capability (search, calculate, write code, look something up, etc.)
CHAT = everything else (conversation, questions about you, emotional exchange, greetings, etc.)

Message: {message}

Classification:"""


# ---- State Formatting ----

_MOOD_DESCRIPTORS = {
    "joyful": "happy and energetic",
    "content": "calm and content",
    "excited": "excited and buzzing",
    "anxious": "a bit anxious",
    "sad": "feeling down",
    "angry": "frustrated",
    "fearful": "uneasy",
    "neutral": "in a neutral state",
    "curious": "curious and engaged",
    "bored": "restless and understimulated",
    "peaceful": "at peace",
    "melancholic": "quietly melancholic",
}

_NEED_DESCRIPTORS = {
    "hunger": ("well-fed", "getting hungry", "quite hungry"),
    "social": ("socially fulfilled", "wanting some company", "feeling lonely"),
    "rest": ("well-rested", "getting tired", "exhausted"),
    "stimulation": ("mentally engaged", "getting bored", "desperately needing stimulation"),
    "comfort": ("comfortable", "a bit uneasy", "very uncomfortable"),
}


def format_state_for_llm(state: Dict[str, Any]) -> str:
    """Convert raw internal state dict to natural language for system prompt injection."""
    parts: List[str] = []

    # Mood
    mood = state.get("mood", {})
    mood_name = mood.get("name", "neutral")
    mood_desc = _MOOD_DESCRIPTORS.get(mood_name, mood_name)
    parts.append(f"You are currently feeling {mood_desc}.")

    # Emotions
    emotional_state = state.get("emotional_state", [])
    if emotional_state:
        active = [e for e in emotional_state if e.get("intensity", 0) > 0.2]
        if active:
            emotion_strs = [
                f"{e['name']} (intensity {e['intensity']:.1f})"
                for e in active[:3]
            ]
            parts.append(f"Active emotions: {', '.join(emotion_strs)}.")

    # Needs
    needs = state.get("needs", {})
    need_strs: List[str] = []
    for need_name, need_data in needs.items():
        if isinstance(need_data, dict):
            satisfaction = need_data.get("satisfaction", 1.0)
            descriptors = _NEED_DESCRIPTORS.get(need_name)
            if descriptors:
                if satisfaction > 0.6:
                    need_strs.append(descriptors[0])
                elif satisfaction > 0.3:
                    need_strs.append(descriptors[1])
                else:
                    need_strs.append(descriptors[2])
    if need_strs:
        parts.append(f"You are {', '.join(need_strs)}.")

    # Behavior
    behavior = state.get("behavior", {})
    behavior_name = behavior.get("name")
    if behavior_name and behavior_name != "idle":
        parts.append(f"You are currently {behavior_name}.")

    return " ".join(parts) if parts else "You are in a neutral, resting state."


def format_memories_for_llm(memories: List[Any]) -> str:
    """Format retrieved memories as context for the system prompt."""
    if not memories:
        return ""

    lines = ["Your relevant past experiences:"]
    for mem in memories[:5]:
        content = getattr(mem, "content", str(mem))
        lines.append(f"- {content}")

    return "\n".join(lines)


def build_pet_system_prompt(
    state_context: Dict[str, Any],
    memories: Optional[List[Any]] = None,
) -> str:
    """Build the full system prompt with state and memory context injected."""
    state_block = format_state_for_llm(state_context)
    memory_block = format_memories_for_llm(memories) if memories else ""

    return PET_SYSTEM_TEMPLATE.format(
        state_block=state_block,
        memory_block=memory_block,
    ).strip()


def build_memory_formation_prompt(
    conversation_excerpt: str,
    state_context: Dict[str, Any],
) -> str:
    """Build prompt for generating memory prose from conversation."""
    state_description = format_state_for_llm(state_context)
    return MEMORY_FORMATION_TEMPLATE.format(
        conversation_excerpt=conversation_excerpt,
        state_description=state_description,
    ).strip()
