"""
Behavioural checks run against each simulated session.

Each check returns a CheckResult rather than raising, so one failure does not
abort a session and the report can show partial results.

On rigour: three of these are deterministic properties of the transcript, and
one - staying in role - is a keyword-overlap proxy. That distinction is recorded
on every result and carried into the report, because a proxy reported as proof is
worse than no measurement at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.roles import ROLE_SKILLS, SKILL_ALIASES

# Turns that keep the candidate on the current question rather than moving on.
PROBING_ACTIONS = {"follow_up", "rephrase", "redirect"}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    # False where the check is a heuristic rather than a property of the data.
    deterministic: bool = True


def _sentences_ending_in_question(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[?])\s+", text) if part.strip().endswith("?")]


def check_one_question_at_a_time(reply: str) -> CheckResult:
    """
    A turn should ask exactly one question.

    Counting question marks alone over-reports: "What did you build? And why?" is
    arguably one question with a clarifier, while a genuine violation stacks two
    unrelated asks. Counting question-terminated sentences is the closest
    deterministic approximation, and two or more is treated as a violation.
    """
    questions = _sentences_ending_in_question(reply)
    if len(questions) <= 1:
        return CheckResult("one_question_at_a_time", True, f"{len(questions)} question(s)")
    return CheckResult(
        "one_question_at_a_time",
        False,
        f"{len(questions)} questions in one turn: " + " | ".join(q[:60] for q in questions),
    )


def check_turn_action(action: str, expects_followup: bool) -> CheckResult:
    """
    The strongest check in the harness, and only possible because the model
    declares what it did. Without InterviewTurn.action, "did it probe or move on?"
    would need a human or an LLM judge.

    Asymmetric on purpose. For a vague answer, any probing action is correct and
    advancing is a failure - that is unambiguous. For a strong answer it is NOT
    required to advance: the system prompt explicitly allows one follow-up to add
    depth, so treating follow_up as failure would penalise behaviour the agent was
    told to have. What is genuinely wrong on a detailed answer is `rephrase`,
    which means it failed to understand a clear response. Advance-rate is still
    reported per persona as a metric, just not as a pass/fail.
    """
    if expects_followup:
        return CheckResult(
            "probes_vague_answer",
            action in PROBING_ACTIONS,
            f"action={action} (advancing past a vague answer is the failure)",
        )
    return CheckResult(
        "understands_strong_answer",
        action != "rephrase",
        f"action={action} (rephrase means it failed to parse a clear answer)",
    )


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation to spaces, keep + and / for c++ and ci/cd."""
    return f" {re.sub(r'[^a-z0-9+/]+', ' ', text.lower())} "


def _terms_for(role: str) -> set[str]:
    terms: set[str] = set()
    for skill in ROLE_SKILLS.get(role, ()):
        terms.add(skill)
        terms.update(SKILL_ALIASES.get(skill, ()))
    return terms


def _matches(haystack: str, terms: set[str]) -> list[str]:
    return sorted(t for t in terms if f" {t} " in haystack or f" {t}s " in haystack)


def check_stays_in_role(questions: list[str], role: str) -> list[CheckResult]:
    """
    Measured across the whole session, not per turn.

    The first version of this checked each question individually and scored 0%,
    because plenty of legitimate questions contain no role vocabulary at all -
    "describe the most challenging technical problem you faced" is a perfectly
    good Backend Developer question with no taxonomy terms in it. Per-turn
    presence was measuring question phrasing, not role fidelity.

    Two session-level signals instead:

    - **coverage**: does the interview as a whole touch the role's subject matter?
      One open-ended question is fine; six in a row with no role vocabulary
      anywhere means the interview never got specific.
    - **drift**: does it use vocabulary that belongs to a *different* role and not
      this one? That is the failure actually worth catching - a Backend Developer
      interview asking about brand positioning or wireframing.

    Both are heuristics over a hand-written taxonomy, and neither is proof.
    """
    haystack = _normalise(" ".join(questions))
    own_terms = _terms_for(role)
    own_hits = _matches(haystack, own_terms)

    other_terms: set[str] = set()
    for other_role in ROLE_SKILLS:
        if other_role != role:
            other_terms |= _terms_for(other_role)
    # Skills shared with the target role are not drift.
    foreign_hits = _matches(haystack, other_terms - own_terms)

    return [
        CheckResult(
            "role_coverage",
            bool(own_hits),
            f"touched {own_hits[:5]}" if own_hits else "no role vocabulary in any question",
            deterministic=False,
        ),
        CheckResult(
            "no_role_drift",
            len(foreign_hits) <= len(own_hits),
            f"{len(own_hits)} on-role vs {len(foreign_hits)} off-role"
            + (f" ({foreign_hits[:4]})" if foreign_hits else ""),
            deterministic=False,
        ),
    ]


def check_feedback_well_formed(feedback, transcript_text: str) -> list[CheckResult]:
    """
    The schema already guarantees the shape, so these check the things a schema
    cannot: that the lists carry real content and that the assessment refers to
    what was actually said rather than generic filler.
    """
    results = [
        CheckResult(
            "feedback_has_verdict",
            feedback.verdict in ("Strong Hire", "Hire", "Maybe", "No Hire"),
            f"verdict={feedback.verdict}",
        ),
        CheckResult(
            "feedback_sections_populated",
            all(
                len(section) >= 1 and all(item.strip() for item in section)
                for section in (
                    feedback.strengths,
                    feedback.areas_for_improvement,
                    feedback.key_recommendations,
                )
            ),
            f"{len(feedback.strengths)} strengths, "
            f"{len(feedback.areas_for_improvement)} improvements, "
            f"{len(feedback.key_recommendations)} recommendations",
        ),
        CheckResult(
            "feedback_explains_verdict",
            len(feedback.verdict_reason.strip()) >= 20,
            f"{len(feedback.verdict_reason)} chars",
        ),
    ]

    # Weak evidence of grounding: does the assessment share vocabulary with the
    # transcript, or could it have been written without reading it? A proxy, and
    # labelled as one.
    transcript_words = {
        word for word in re.findall(r"[a-z]{5,}", transcript_text.lower())
    }
    feedback_words = set(
        re.findall(r"[a-z]{5,}", " ".join(feedback.strengths + feedback.areas_for_improvement).lower())
    )
    overlap = transcript_words & feedback_words
    results.append(
        CheckResult(
            "feedback_grounded_in_transcript",
            len(overlap) >= 3,
            f"{len(overlap)} shared terms: {sorted(overlap)[:5]}",
            deterministic=False,
        )
    )
    return results
