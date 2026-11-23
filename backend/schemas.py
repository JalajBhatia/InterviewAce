"""
Typed model outputs.

Keeping these in their own module rather than in main.py: they describe what the
LLM must return, not what an HTTP endpoint accepts, and later phases (resume
scoring) need their own schemas alongside these.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .roles import SUPPORTED_ROLES

# Built from the taxonomy rather than retyped, so the roles the model may return
# cannot drift from the roles the app supports. Because this is the schema the
# model is bound to, an unsupported role is not unlikely - it is impossible.
RoleName = Literal[SUPPORTED_ROLES]

ExperienceLevel = Literal["Junior", "Mid-Level", "Senior"]


class RoleDetection(BaseModel):
    """What the model concluded about the target role, before the user confirms."""

    role: RoleName = Field(
        description="The single best-matching role from the supported list."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high: the source names the role or is unambiguous. "
            "medium: strongly implied by the skills present. "
            "low: a guess from thin or conflicting evidence."
        )
    )
    reasoning: str = Field(
        description="One short sentence citing the evidence, shown to the user."
    )
    skills_found: list[str] = Field(
        default_factory=list,
        description="Concrete skills, tools and technologies named in the source.",
    )
    suggested_experience_level: ExperienceLevel | None = Field(
        default=None,
        description="Seniority implied by the source, or null if it does not say.",
    )

# Fixed set on purpose. A free-text verdict cannot be counted, compared across
# sessions, or asserted in the eval harness - "Strong Hire", "strong hire!" and
# "I would lean towards hiring" would all be distinct values.
Verdict = Literal["Strong Hire", "Hire", "Maybe", "No Hire"]

InterviewType = Literal["Technical", "HR"]
DifficultyLevel = Literal["Easy", "Medium", "Hard"]

# --- Rubric Scoring (Milestone 1 / Problem C10) ----------------------------

class RubricScores(BaseModel):
    """
    Fixed 4-criteria evaluation rubric (0-5 scale each, total 20).
    Backend validates bounds (0-5) and strictly recomputes total and percentage
    rather than blindly trusting the LLM.
    """
    correctness: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Factual and technical correctness of concepts (0–5).",
    )
    relevance: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Direct relevance to the question asked without evasion (0–5).",
    )
    clarity: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Structure, logical flow, and articulation of the answer (0–5).",
    )
    completeness: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Depth, coverage of key points, edge cases, and tradeoffs (0–5).",
    )
    total_score: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Sum of 4 criteria (0–20), recomputed server-side.",
    )
    percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Calculated as (total_score / 20) * 100, recomputed server-side.",
    )


class AnswerEvaluation(BaseModel):
    """Evaluation of a single candidate answer against the fixed rubric."""

    rubric: RubricScores
    strengths: list[str] = Field(
        default_factory=list,
        description="Key strengths demonstrated in this specific answer.",
    )
    areas_for_improvement: list[str] = Field(
        default_factory=list,
        description="Constructive and actionable areas for improvement.",
    )
    suggested_improved_answer: str = Field(
        default="",
        description="Exemplar model answer showing how the candidate could improve.",
    )
    follow_up_question: str = Field(
        default="",
        description="A relevant follow-up question or probe based on this answer.",
    )
    question_number: Optional[int] = Field(
        default=None,
        description="1-based index of the interview question evaluated.",
    )
    paste_detected: bool = Field(
        default=False,
        description="Whether a paste event was detected for this answer.",
    )


# What the interviewer did with this turn. Making the model declare it is the
# only way to count *questions* rather than *turns*: a follow-up and a new
# question are indistinguishable from the outside, so the previous code counted
# every user message and a candidate who needed three clarifications burned four
# of their fifteen question slots.
TurnAction = Literal["advance", "follow_up", "rephrase", "redirect"]


class InterviewTurn(BaseModel):
    """One interviewer response, plus what kind of response it is and optional answer evaluation."""

    action: TurnAction = Field(
        description=(
            "advance: the previous answer was complete, so this is a NEW question. "
            "follow_up: probing deeper on the same topic. "
            "rephrase: the answer was irrelevant or unintelligible, so ask again differently. "
            "redirect: the candidate went off-topic or out of bounds, so steer back."
        )
    )
    reply: str = Field(
        description="What to say to the candidate. Exactly one question. No prefixes or labels."
    )
    evaluation: AnswerEvaluation | None = Field(
        default=None,
        description="Rubric-based evaluation of the candidate's preceding answer.",
    )


class PerformanceSummary(BaseModel):
    """
    Final performance report synthesized from actual session evaluations.
    """
    questions_answered: int
    total_marks_obtained: int
    max_possible_marks: int
    overall_percentage: float
    average_score: float
    strongest_criterion: str
    weakest_criterion: str
    criterion_totals: dict[str, int] = Field(default_factory=dict)
    criterion_averages: dict[str, float] = Field(default_factory=dict)
    overall_strengths: list[str] = Field(default_factory=list)
    areas_that_need_improvement: list[str] = Field(default_factory=list)
    personalized_recommendations: list[str] = Field(default_factory=list)


class InterviewFeedback(BaseModel):
    """
    Structured assessment produced at the end of an interview.

    Previously the model was asked to emit markdown headings and the result was
    returned as one opaque string. Nothing verified the headings were present, so
    "structured feedback" held only by convention - a reworded response silently
    produced an unparseable blob. Binding the response to this schema makes the
    shape enforced rather than requested: a missing field or an invented verdict
    is a validation error, not a rendering bug discovered by a user.
    """

    overall_performance: str = Field(
        description="One or two sentences summarising how the candidate performed overall."
    )
    strengths: list[str] = Field(
        description="Specific things the candidate did well, each citing something they actually said.",
        min_length=1,
    )
    areas_for_improvement: list[str] = Field(
        description="Specific weaknesses, each paired with actionable advice.",
        min_length=1,
    )
    key_recommendations: list[str] = Field(
        description="Concrete next steps the candidate should take before a real interview.",
        min_length=1,
    )
    verdict: Verdict = Field(
        description="Overall hiring recommendation, judged against the expected seniority."
    )
    verdict_reason: str = Field(
        description="One sentence explaining the verdict."
    )
    performance_summary: PerformanceSummary | None = Field(
        default=None,
        description="Rubric score breakdown and analytics from the actual session.",
    )


# --- Resume scoring -------------------------------------------------------

ClaimStatus = Literal["supported", "unsupported", "undersold"]


class ClaimCheck(BaseModel):
    """
    One resume claim held up against what the candidate actually said.

    This is the part no resume scanner can do, because it needs both documents:
    a scanner has the resume and nothing else. Kept as discrete per-claim verdicts
    with a citation rather than a score, so every judgement can be traced back to
    something in the transcript.
    """

    claim: str = Field(description="The skill or claim as the resume states it.")
    status: ClaimStatus = Field(
        description=(
            "supported: the interview backed it up. "
            "unsupported: the resume claims it but the answers did not demonstrate it. "
            "undersold: demonstrated well in the interview but weak or absent on the resume."
        )
    )
    evidence: str = Field(
        description="Short quote or paraphrase from the transcript justifying this."
    )


class ClaimAnalysis(BaseModel):
    """Cross-reference of resume claims against interview evidence."""

    checks: list[ClaimCheck] = Field(default_factory=list)
    summary: str = Field(description="One or two sentences on the overall picture.")


class ScoreFinding(BaseModel):
    code: str
    severity: Literal["critical", "warning", "suggestion"]
    message: str
    points_deducted: int
    evidence: str = ""


class ScoreCategory(BaseModel):
    name: str
    score: int
    max_score: int
    findings: list[ScoreFinding] = Field(default_factory=list)


class RoleMatch(BaseModel):
    target: str
    coverage: float
    matched: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class ResumeScore(BaseModel):
    """
    The complete assessment.

    ``total`` is produced entirely by deterministic rules - the same resume always
    scores the same. ``narrative`` is the only LLM-written field, and it is
    constrained to the findings already computed: it phrases them, it never adds
    one and never moves the number.
    """

    total: int
    categories: list[ScoreCategory]
    role_match: RoleMatch
    claim_analysis: ClaimAnalysis | None = None
    narrative: str | None = None
