"""The scoring rubric: deterministic, additive, traceable."""

import pytest

from backend.resume_analysis import analyze_resume
from backend.resume_score import MATCH_MAX, PARSE_MAX, QUALITY_MAX, score_resume
from tests.test_resume_checks import GOOD

BAD = "John\nDid some stuff at a company. Worked on things."


def test_weights_sum_to_one_hundred():
    assert PARSE_MAX + MATCH_MAX + QUALITY_MAX == 100


def test_parse_compatibility_is_weighted_highest():
    """A document that cannot be read is broken as a fact; style is taste."""
    assert PARSE_MAX > MATCH_MAX > QUALITY_MAX


def test_scoring_is_deterministic():
    analysis = analyze_resume(GOOD)
    first = score_resume(analysis, "Backend Developer")
    second = score_resume(analyze_resume(GOOD), "Backend Developer")
    assert first.total == second.total
    assert first.model_dump() == second.model_dump()


def test_total_is_the_sum_of_categories():
    score = score_resume(analyze_resume(GOOD), "Backend Developer")
    assert score.total == sum(c.score for c in score.categories)


def test_every_deduction_traces_to_a_finding():
    score = score_resume(analyze_resume(BAD), "Backend Developer")
    for category in score.categories:
        lost = category.max_score - category.score
        explained = sum(f.points_deducted for f in category.findings)
        # Deductions are capped at zero, so explained may exceed what was lost.
        assert explained >= lost, category.name


def test_scores_stay_in_range():
    for text in (GOOD, BAD, "", "x" * 5000):
        score = score_resume(analyze_resume(text), "Backend Developer")
        assert 0 <= score.total <= 100
        for category in score.categories:
            assert 0 <= category.score <= category.max_score


def test_a_good_resume_outscores_a_bad_one():
    good = score_resume(analyze_resume(GOOD), "Backend Developer").total
    bad = score_resume(analyze_resume(BAD), "Backend Developer").total
    assert good > bad + 25


def test_job_description_becomes_the_target_when_supplied():
    jd = "Backend Engineer with Python, PostgreSQL, Kubernetes and Terraform."
    score = score_resume(analyze_resume(GOOD), "Backend Developer", jd)
    assert "job description" in score.role_match.target


def test_role_taxonomy_is_the_fallback():
    score = score_resume(analyze_resume(GOOD), "Backend Developer")
    assert "curated" in score.role_match.target


def test_blank_job_description_falls_back_to_the_taxonomy():
    score = score_resume(analyze_resume(GOOD), "Backend Developer", "   ")
    assert "curated" in score.role_match.target


def test_matching_the_wrong_role_lowers_the_match_category():
    backend = score_resume(analyze_resume(GOOD), "Backend Developer")
    designer = score_resume(analyze_resume(GOOD), "UX/UI Designer")
    by_name = lambda s: {c.name: c.score for c in s.categories}
    assert by_name(backend)["Role match"] > by_name(designer)["Role match"]


def test_parse_failures_dominate_a_stylistically_perfect_resume():
    """The weighting has to say that unreadable beats untidy."""
    analysis = analyze_resume(GOOD)
    analysis.warnings.append("2 of 2 page(s) contained no readable text.")
    scored = score_resume(analysis, "Backend Developer")
    parse = next(c for c in scored.categories if c.name == "Parse compatibility")
    assert parse.score < PARSE_MAX
    assert any(f.severity == "critical" for f in parse.findings)


def test_missing_terms_are_reported_but_capped():
    score = score_resume(analyze_resume(BAD), "Backend Developer")
    assert len(score.role_match.missing) <= 12, "a wall of gaps is not actionable"


def test_no_llm_fields_are_populated_by_the_scorer():
    """The rubric produces the number; the model only writes prose afterwards."""
    score = score_resume(analyze_resume(GOOD), "Backend Developer")
    assert score.narrative is None
    assert score.claim_analysis is None


# --- claim cross-reference filtering --------------------------------------

def test_absence_based_verdicts_are_dropped(monkeypatch):
    """
    A claim the interview never probed is not evidence of a gap. The prompt says
    so, but the rule is enforced in code because a drifted verdict becomes an
    accusation the transcript does not support.
    """
    from backend import resume_review
    from backend.schemas import ClaimAnalysis, ClaimCheck

    fake = ClaimAnalysis(
        summary="mixed",
        checks=[
            ClaimCheck(claim="Kubernetes", status="unsupported",
                       evidence="The candidate did not mention Kubernetes."),
            ClaimCheck(claim="Kafka", status="unsupported",
                       evidence="Not discussed during the interview."),
            ClaimCheck(claim="PostgreSQL", status="supported",
                       evidence="I designed the partitioning strategy and tuned autovacuum."),
        ],
    )
    monkeypatch.setattr(resume_review, "generate_structured", lambda *a, **k: fake)

    result = resume_review.analyse_claims("resume", [{"role": "user", "content": "hi"}])
    assert [c.claim for c in result.checks] == ["PostgreSQL"]


def test_evidence_backed_verdicts_survive(monkeypatch):
    from backend import resume_review
    from backend.schemas import ClaimAnalysis, ClaimCheck

    fake = ClaimAnalysis(
        summary="s",
        checks=[
            ClaimCheck(claim="Kubernetes", status="unsupported",
                       evidence="The platform team owned that, I just wrote the YAML."),
        ],
    )
    monkeypatch.setattr(resume_review, "generate_structured", lambda *a, **k: fake)

    result = resume_review.analyse_claims("resume", [{"role": "user", "content": "hi"}])
    assert len(result.checks) == 1, "a real quoted gap must be kept"
