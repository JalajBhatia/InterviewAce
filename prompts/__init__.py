"""
InterviewAce Prompt Engineering System
Prompt versions registry for academic evaluation and production use.
"""

from .v1_baseline import V1_SYSTEM_PROMPT, V1_EVALUATION_INSTRUCTIONS
from .v2_structured import V2_SYSTEM_PROMPT, V2_EVALUATION_INSTRUCTIONS
from .v3_improved import V3_SYSTEM_PROMPT, V3_EVALUATION_INSTRUCTIONS
from .v4_final import V4_SYSTEM_PROMPT, V4_EVALUATION_INSTRUCTIONS, V4_FEEDBACK_SYSTEM_PROMPT

PROMPT_VERSIONS = {
    "v1": {
        "name": "V1 — Baseline Prompt",
        "description": "Simple interview assistant prompt asking questions with basic conversational feedback.",
        "system_prompt": V1_SYSTEM_PROMPT,
        "evaluation_prompt": V1_EVALUATION_INSTRUCTIONS,
    },
    "v2": {
        "name": "V2 — Structured Prompt",
        "description": "Adds 4-criteria rubric (Correctness, Relevance, Clarity, Completeness) and structured feedback fields.",
        "system_prompt": V2_SYSTEM_PROMPT,
        "evaluation_prompt": V2_EVALUATION_INSTRUCTIONS,
    },
    "v3": {
        "name": "V3 — Improved Prompt",
        "description": "Adds strict score anchoring (0-5 rubric definitions), output constraints, anti-hallucination, and follow-up guidance.",
        "system_prompt": V3_SYSTEM_PROMPT,
        "evaluation_prompt": V3_EVALUATION_INSTRUCTIONS,
    },
    "v4": {
        "name": "V4 — Final Production Prompt",
        "description": "Combines role-specific interview dimensions, difficulty adaptation, exemplar answer modeling, and placement-readiness coaching.",
        "system_prompt": V4_SYSTEM_PROMPT,
        "evaluation_prompt": V4_EVALUATION_INSTRUCTIONS,
    },
}

DEFAULT_PROMPT_VERSION = "v4"

def get_prompt_version(version_key: str = "v4") -> dict:
    normalized = (version_key or "v4").lower().strip()
    return PROMPT_VERSIONS.get(normalized, PROMPT_VERSIONS[DEFAULT_PROMPT_VERSION])
