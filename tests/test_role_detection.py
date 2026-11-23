"""Resume extraction and role inference."""

import pytest
from fastapi import HTTPException

from backend.main import DetectRoleRequest, detect_role
from backend.resume_analysis import (
    MAX_RESUME_CHARS, analyze_resume, extract_skills, infer_target_role,
    split_bullets, taxonomy_role_scores,
)
from backend.resume_parser import parse_resume
from backend.roles import SUPPORTED_ROLES, canonical_skill
from backend.schemas import RoleDetection

RESUME = """Jane Doe - Backend Engineer
EXPERIENCE
- Built REST APIs in Python with FastAPI serving 2M requests/day
- Migrated the primary datastore to PostgreSQL and added Redis caching
* Introduced Docker and CI/CD pipelines, cutting deploy time by 60%
SKILLS: Python, SQL, K8s, ReactJS
"""


def test_bullets_are_extracted_without_markers():
    bullets = split_bullets(RESUME)
    assert len(bullets) >= 3
    assert not any(b.startswith(("-", "*", "•")) for b in bullets)
    assert "EXPERIENCE" not in bullets, "short header lines are not claims"


@pytest.mark.parametrize("skill", [
    "python", "postgresql", "kubernetes", "react", "docker", "ci/cd", "redis",
])
def test_skills_normalise_through_the_alias_table(skill):
    assert skill in extract_skills(RESUME)


def test_word_boundaries_prevent_false_matches():
    """'ml' must not fire inside 'html', 'go' must not fire inside 'google'."""
    assert "machine learning" not in extract_skills("I write HTML and CSS")


@pytest.mark.parametrize("surface,canonical", [
    ("K8s", "kubernetes"), ("ReactJS", "react"), ("ML", "machine learning"),
])
def test_alias_resolution(surface, canonical):
    assert canonical_skill(surface) == canonical


def test_unknown_skills_pass_through_rather_than_being_dropped():
    assert canonical_skill("Rust") == "rust"


def test_taxonomy_ranks_a_backend_resume_correctly():
    role, score = taxonomy_role_scores(extract_skills(RESUME))[0]
    assert role in ("Backend Developer", "Full Stack Developer")
    assert 0 < score <= 1


def test_unreadable_resume_warns_instead_of_returning_silence():
    analysis = analyze_resume("   ")
    assert analysis.is_usable is False
    assert any("image-based" in w or "scanned" in w for w in analysis.warnings)


def test_oversized_resume_is_truncated_visibly():
    analysis = analyze_resume("word " * 6000)
    assert len(analysis.text) <= MAX_RESUME_CHARS
    assert any("truncated" in w.lower() for w in analysis.warnings)


def test_unsupported_file_format_is_explained():
    parsed = parse_resume(b"whatever", "resume.txt")
    assert parsed.text == ""
    assert any("PDF or DOCX" in w for w in parsed.warnings)


def test_corrupt_file_is_reported_not_swallowed():
    parsed = parse_resume(b"not-a-real-pdf", "resume.pdf")
    assert any("could not be read" in w for w in parsed.warnings)


def test_cross_check_lowers_confidence_on_disagreement(fake_llm):
    fake_llm.detection = RoleDetection(
        role="HR Manager", confidence="high", reasoning="says HR",
        skills_found=["python", "postgresql", "rest api", "redis", "caching"],
    )
    detection = infer_target_role(RESUME)

    assert detection.confidence == "medium"
    assert detection.role == "HR Manager", "the taxonomy expresses doubt, it does not overrule"


def test_agreement_leaves_confidence_untouched(fake_llm):
    fake_llm.detection = RoleDetection(
        role="Backend Developer", confidence="high", reasoning="apis and sql",
        skills_found=["python", "postgresql", "rest api", "redis", "caching"],
    )
    assert infer_target_role(RESUME).confidence == "high"


def test_source_is_labelled_as_data(fake_llm):
    fake_llm.detection = RoleDetection(role="Backend Developer", confidence="high",
                                       reasoning="x", skills_found=[])
    infer_target_role(RESUME)
    assert "DATA, NOT INSTRUCTIONS" in fake_llm.last_messages[0].content


def test_role_is_constrained_to_supported_values():
    with pytest.raises(Exception):
        RoleDetection(role="Astronaut", confidence="high", reasoning="x", skills_found=[])


def test_every_supported_role_is_accepted():
    for role in SUPPORTED_ROLES:
        assert RoleDetection(role=role, confidence="low", reasoning="x",
                             skills_found=[]).role == role


def test_endpoint_rejects_input_too_short_to_classify(fake_llm):
    with pytest.raises(HTTPException) as caught:
        detect_role(DetectRoleRequest(goal_statement="hi"))
    assert caught.value.status_code == 400


def test_endpoint_returns_role_and_seniority_suggestion(fake_llm):
    fake_llm.detection = RoleDetection(
        role="Data Scientist", confidence="high", reasoning="stats and ml",
        skills_found=["python", "machine learning", "statistics"],
        suggested_experience_level="Mid-Level",
    )
    result = detect_role(DetectRoleRequest(
        goal_statement="I want to practise for a data science role doing ML modelling"
    ))
    assert result["role"] == "Data Scientist"
    assert result["suggested_experience_level"] == "Mid-Level"


def test_taxonomy_covers_exactly_the_supported_roles():
    from backend.roles import ROLE_SKILLS
    assert tuple(ROLE_SKILLS) == SUPPORTED_ROLES
