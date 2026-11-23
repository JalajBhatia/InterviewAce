"""
The LLM-written parts of the resume review.

Both functions are deliberately fenced in. The score is computed by rules before
either runs, and neither can change it:

- ``write_narrative`` is given the findings and asked to phrase them. It is told
  it may not invent a finding or dispute the number.
- ``analyse_claims`` is the one place semantic judgement is genuinely required -
  deciding whether "expert in Kubernetes" was borne out by what the candidate
  said needs to understand both. It returns per-claim verdicts with citations
  rather than a score, so each judgement is checkable against the transcript.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from .llm_client import generate_structured, generate_response
from .prompts import (
    CLAIM_ANALYSIS_PROMPT, CLAIM_ANALYSIS_SYSTEM_PROMPT,
    SCORE_NARRATIVE_PROMPT, SCORE_NARRATIVE_SYSTEM_PROMPT,
)
from .schemas import ClaimAnalysis, ResumeScore

# Enough of the transcript to judge claims against, bounded so a long interview
# does not blow up the request.
MAX_TRANSCRIPT_TURNS = 40

# Phrasings that mean "this did not come up" rather than "here is what they said".
# A claim the interview never probed is not evidence of a gap - the candidate was
# simply never asked - so reporting one as unsupported is both wrong and unfair.
_ABSENCE_MARKERS = (
    "did not", "didn't", "not mention", "no mention", "was not asked",
    "wasn't asked", "not discussed", "not covered", "not addressed",
    "no evidence", "did no", "not elaborate", "not provide",
)


def _is_absence_based(evidence: str) -> bool:
    lowered = evidence.lower()
    return any(marker in lowered for marker in _ABSENCE_MARKERS)


def analyse_claims(resume_text: str, history: list[dict]) -> ClaimAnalysis:
    """
    Cross-reference resume claims against interview evidence.

    The feature no resume scanner can offer, because it needs both documents and a
    scanner only has one.
    """
    turns = [
        {
            "speaker": "interviewer" if turn["role"] == "model" else "candidate",
            "text": turn["content"],
        }
        for turn in history[-MAX_TRANSCRIPT_TURNS:]
    ]
    analysis = generate_structured(
        [
            SystemMessage(content=CLAIM_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(
                content=CLAIM_ANALYSIS_PROMPT.format(
                    resume_text=resume_text,
                    transcript_json=json.dumps(turns, indent=2, ensure_ascii=False),
                )
            ),
        ],
        ClaimAnalysis,
    )

    # Enforce the "no absence-based verdicts" rule in code rather than trusting the
    # prompt. The instruction is in the system prompt too, but a rule that decides
    # whether a candidate is told their claim is unsupported is worth guaranteeing
    # structurally: a model that drifts once produces an accusation the transcript
    # does not support.
    analysis.checks = [
        check for check in analysis.checks if not _is_absence_based(check.evidence)
    ]
    return analysis


def write_narrative(score: ResumeScore) -> str:
    """Phrase the already-computed findings. Cannot add one, cannot move the score."""
    categories = "\n".join(
        f"- {c.name}: {c.score}/{c.max_score}" for c in score.categories
    )
    findings = "\n".join(
        f"- [{f.severity}] {f.message}"
        for category in score.categories
        for f in category.findings
    ) or "- No issues found."

    return generate_response(
        SCORE_NARRATIVE_PROMPT.format(
            total=score.total, categories=categories, findings=findings
        ),
        system_instruction=SCORE_NARRATIVE_SYSTEM_PROMPT,
    )
