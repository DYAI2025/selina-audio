"""WTME Marker System wrapper for emotion detection.

Thin wrapper around the marker_pipeline for use in the Selina audio service.
Analyzes text for emotional patterns (ATOs, SEMs) and returns
emotion labels suitable for avatar expressions and TTS instructions.
"""

import sys
from pathlib import Path

# Add marker_system to Python path
_MARKER_DIR = Path("/home/dyai/marker_system")
if str(_MARKER_DIR) not in sys.path:
    sys.path.insert(0, str(_MARKER_DIR))

from pipeline.marker_pipeline import process_messages, detect_atos, compose_sems, load_atos, load_sems

# Pre-load definitions once
_ato_defs = None
_sem_defs = None


def _ensure_loaded():
    global _ato_defs, _sem_defs
    if _ato_defs is None:
        _ato_defs = load_atos()
        _sem_defs = load_sems()
        print(f"[Markers] Loaded {len(_ato_defs)} ATOs, {len(_sem_defs)} SEMs")


# ---------------------------------------------------------------------------
# SEM → Emotion mapping for avatar + TTS
# ---------------------------------------------------------------------------

# Maps SEM names to a simplified emotion label
SEM_TO_EMOTION: dict[str, str] = {
    # Positive
    "SEM_ACTIVE_ENGAGEMENT": "engaged",
    "SEM_PASSIVE_ENGAGEMENT": "engaged",
    "SEM_POSITIVE_MOMENTUM": "happy",
    "SEM_FIRM_PROMISE": "engaged",
    "SEM_IMMEDIATE_ACTION": "engaged",
    # Negative / Conflict
    "SEM_POTENTIAL_ESCALATION": "frustrated",
    "SEM_REPETITIVE_ESCALATION": "frustrated",
    "SEM_CONFLICT_START": "frustrated",
    "SEM_FRUSTRATED": "frustrated",
    "SEM_STRONG_DISAGREEMENT": "frustrated",
    "SEM_REFUSAL": "frustrated",
    "SEM_FIRM_WITHDRAWAL": "sad",
    "SEM_SOFT_WITHDRAWAL": "sad",
    # Uncertainty / Anxiety
    "SEM_HESITANT_UNCERTAINTY": "anxious",
    "SEM_CERTAIN_UNCERTAINTY": "uncertain",
    "SEM_HESITANT_AGREEMENT": "uncertain",
    "SEM_HESITANT_REQUEST": "uncertain",
    "SEM_TIME_CONFLICT": "anxious",
    "SEM_EMOTIONAL_MIXED": "mixed",
}

# Dominant ATO fallback when no SEMs fire
ATO_TO_EMOTION: dict[str, str] = {
    "ATO_POSITIVE": "happy",
    "ATO_NEGATIVE": "sad",
    "ATO_FRUSTRATION": "frustrated",
    "ATO_HESITATION": "anxious",
    "ATO_ENGAGEMENT": "engaged",
    "ATO_WITHDRAWAL": "sad",
    "ATO_INTENSIFIER": "intense",
}

# Emotion → avatar expression (TalkingHead mood / blendshape hints)
EMOTION_TO_AVATAR: dict[str, dict] = {
    "happy":      {"mood": "happy", "expression": "smile"},
    "sad":        {"mood": "sad", "expression": "frown"},
    "frustrated": {"mood": "angry", "expression": "serious"},
    "anxious":    {"mood": "concerned", "expression": "worried"},
    "uncertain":  {"mood": "neutral", "expression": "thinking"},
    "engaged":    {"mood": "happy", "expression": "attentive"},
    "mixed":      {"mood": "neutral", "expression": "thoughtful"},
    "intense":    {"mood": "surprised", "expression": "alert"},
    "neutral":    {"mood": "neutral", "expression": "idle"},
}

# Emotion → TTS voice instruction (German, for Qwen3-TTS instruct param)
EMOTION_TO_TTS: dict[str, str] = {
    "happy":      "Mit fröhlicher, warmer Stimme",
    "sad":        "Mit sanfter, einfühlsamer Stimme",
    "frustrated": "Mit ruhiger, neutraler Stimme",
    "anxious":    "Mit ruhiger, beruhigender Stimme",
    "uncertain":  "Mit geduldiger, ermutigender Stimme",
    "engaged":    "Mit enthusiastischer, lebhafter Stimme",
    "mixed":      "Mit verständnisvoller Stimme",
    "intense":    "Mit klarer, bestimmter Stimme",
    "neutral":    "",
}


def analyze_single(text: str, sender: str = "user") -> dict:
    """Analyze a single message for emotional markers.

    Returns a compact result with:
    - emotion: dominant emotion label (str)
    - atos: list of detected ATO names
    - sems: list of detected SEM names
    - avatar: mood/expression hints for the avatar
    - tts_instruct: German voice instruction for TTS
    """
    _ensure_loaded()

    # Detect ATOs in the text
    atos_found = detect_atos(text, _ato_defs)
    ato_names = list({a["ato"] for a in atos_found})

    # Compose SEMs from ATOs (needs message-level grouping)
    sems_found = compose_sems(atos_found, _sem_defs)
    sem_names = list({s["sem"] for s in sems_found})

    # Determine dominant emotion from SEMs first, then ATOs as fallback
    emotion = "neutral"
    for sem in sem_names:
        if sem in SEM_TO_EMOTION:
            emotion = SEM_TO_EMOTION[sem]
            break
    if emotion == "neutral" and ato_names:
        # Check ATOs in priority order (content emotions before modifiers)
        for ato in ("ATO_POSITIVE", "ATO_NEGATIVE", "ATO_FRUSTRATION",
                     "ATO_WITHDRAWAL", "ATO_ENGAGEMENT", "ATO_HESITATION",
                     "ATO_INTENSIFIER"):
            if ato in ato_names and ato in ATO_TO_EMOTION:
                emotion = ATO_TO_EMOTION[ato]
                break

    return {
        "emotion": emotion,
        "atos": ato_names,
        "sems": sem_names,
        "avatar": EMOTION_TO_AVATAR.get(emotion, EMOTION_TO_AVATAR["neutral"]),
        "tts_instruct": EMOTION_TO_TTS.get(emotion, ""),
    }


def analyze_conversation(messages: list[dict]) -> dict:
    """Analyze a full conversation through the complete marker pipeline.

    Input: list of {"text": str, "from": str, "timestamp": str}
    Returns the full pipeline result including CLUs and MEMAs.
    """
    _ensure_loaded()
    return process_messages(messages)
