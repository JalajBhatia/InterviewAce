"""
Assembles the deterministic checks into a single 0-100 score.

The arithmetic is the point. Each category starts at its maximum and loses the
points attached to whatever findings fired, so any number on screen can be traced
back to named rules. Nothing here calls an LLM; the narrative is generated
separately and is not allowed to alter the score.

Weighting reflects how objective each signal is:

    parse compatibility   40   a document that cannot be read is broken as a fact
    role match            35   measurable coverage against a stated target
    content quality       25   conventions - real, but a matter of taste

That ordering matters. A resume with beautiful bullets that no parser can read is
in far worse shape than a plainly written one that extracts cleanly, and the
weights have to say so.
"""

from __future__ import annotations

from .matching import MatchResult, match_against_job_description, match_against_role
from .resume_analysis import ResumeAnalysis
from .resume_checks import CheckOutcome, check_content_quality, check_parse_compatibility
from .schemas import ResumeScore, RoleMatch, ScoreCategory, ScoreFinding

PARSE_MAX = 40
MATCH_MAX = 35
QUALITY_MAX = 25

# How many missing terms to show. A wall of sixty gaps is not actionable; the
# highest-information ones are what the candidate should add first.
MAX_REPORTED = 12


def _category(name: str, maximum: int, outcome: CheckOutcome) -> ScoreCategory:
    return ScoreCategory(
        name=name,
        score=max(0, maximum - outcome.deduction()),
        max_score=maximum,
        findings=[
            ScoreFinding(
                code=f.code,
                severity=f.severity,
                message=f.message,
                points_deducted=f.points,
                evidence=f.evidence,
            )
            for f in outcome.findings
        ],
    )


def score_resume(
    analysis: ResumeAnalysis,
    role: str,
    job_description: str | None = None,
) -> ResumeScore:
    """
    Score a parsed resume against a role, or against a pasted job description.

    A real posting is the better target - it is what an applicant tracking system
    actually matches against - so it wins when supplied, with the curated role
    taxonomy as the fallback that always works.
    """
    parse = _category(
        "Parse compatibility",
        PARSE_MAX,
        check_parse_compatibility(analysis.text, analysis.warnings, analysis.bullets),
    )
    quality = _category(
        "Content quality",
        QUALITY_MAX,
        check_content_quality(analysis.bullets, analysis.text),
    )

    match: MatchResult = (
        match_against_job_description(analysis.text, job_description)
        if job_description and job_description.strip()
        else match_against_role(analysis.text, role)
    )

    match_category = ScoreCategory(
        name="Role match",
        score=round(match.score * MATCH_MAX),
        max_score=MATCH_MAX,
        findings=(
            [
                ScoreFinding(
                    code="missing_terms",
                    severity="warning" if match.score < 0.5 else "suggestion",
                    message=(
                        "These terms appear in the target but not in your resume: "
                        + ", ".join(match.missing[:MAX_REPORTED])
                        + ". Recruiter search is exact-term, so a synonym will not "
                        "surface you - use the wording the posting uses, where it is "
                        "genuinely true of your experience."
                    ),
                    points_deducted=MATCH_MAX - round(match.score * MATCH_MAX),
                    evidence=f"{len(match.matched)} matched, {len(match.missing)} missing",
                )
            ]
            if match.missing
            else []
        ),
    )

    categories = [parse, match_category, quality]
    return ResumeScore(
        total=sum(category.score for category in categories),
        categories=categories,
        role_match=RoleMatch(
            target=match.target_description,
            coverage=match.score,
            # Ranked by BM25, so the most informative matches lead.
            matched=[term.term for term in match.matched[:MAX_REPORTED]],
            missing=match.missing[:MAX_REPORTED],
        ),
    )
