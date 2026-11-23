"""Deterministic resume checks: parse compatibility and content quality."""

import pytest

from backend.resume_analysis import split_bullets
from backend.resume_checks import check_content_quality, check_parse_compatibility

GOOD = """Jane Doe
jane.doe@example.com | +44 7700 900123 | linkedin.com/in/janedoe

EXPERIENCE
Senior Backend Engineer, Acme Payments          Jan 2022 - Present
- Built a FastAPI service handling 2 million requests per day
- Reduced p95 latency from 400ms to 90ms by adding a composite index
- Led migration of 12 microservices to Kubernetes, cutting deploy time 60%
- Mentored three junior engineers through their first on-call rotations

Backend Engineer, Betamax Ltd                   2019 - 2021
- Developed the billing reconciliation pipeline processing $4M monthly
- Automated release verification, removing 6 hours of manual work per week

EDUCATION
BSc Computer Science, University of Leeds       2015 - 2019

SKILLS
Python, PostgreSQL, Redis, Docker, Kubernetes, CI/CD
"""


def codes(outcome):
    return {f.code for f in outcome.findings}


def test_a_well_formed_resume_produces_no_critical_findings():
    outcome = check_parse_compatibility(GOOD, [], split_bullets(GOOD))
    criticals = [f for f in outcome.findings if f.severity == "critical"]
    assert not criticals, [f.code for f in criticals]


def test_scanned_pdf_warning_becomes_a_critical_finding():
    outcome = check_parse_compatibility(
        GOOD, ["2 of 2 page(s) contained no readable text."], split_bullets(GOOD)
    )
    assert "no_text_layer" in codes(outcome)
    assert any(f.severity == "critical" for f in outcome.findings)


def test_table_warning_becomes_a_finding():
    outcome = check_parse_compatibility(
        GOOD, ["Document contains 3 table(s)."], split_bullets(GOOD)
    )
    assert "uses_tables" in codes(outcome)


def test_missing_email_is_critical():
    outcome = check_parse_compatibility(
        GOOD.replace("jane.doe@example.com | ", ""), [], split_bullets(GOOD)
    )
    assert "no_email" in codes(outcome)


def test_nonstandard_section_headings_are_flagged():
    creative = GOOD.replace("EXPERIENCE", "MY JOURNEY").replace("EDUCATION", "LEARNING")
    outcome = check_parse_compatibility(creative, [], split_bullets(creative))
    assert "nonstandard_sections" in codes(outcome)


@pytest.mark.parametrize("dates", ["Jan 2022 - Present", "2019 - 2021", "2015 – 2019"])
def test_common_date_formats_are_recognised(dates):
    text = f"EXPERIENCE\nEDUCATION\nSKILLS\na@b.co\nEngineer {dates}\n" + "x " * 300
    outcome = check_parse_compatibility(text, [], ["line"] * 10)
    assert "no_parseable_dates" not in codes(outcome)


def test_missing_dates_are_flagged():
    text = "EXPERIENCE\nEDUCATION\nSKILLS\na@b.co\nEngineer at Acme\n" + "x " * 300
    assert "no_parseable_dates" in codes(check_parse_compatibility(text, [], ["l"] * 10))


def test_long_text_with_few_lines_suggests_multicolumn():
    """The symptom of a two-column layout read across instead of down."""
    text = "EXPERIENCE EDUCATION SKILLS a@b.co 2019 - 2021 " + "word " * 500
    outcome = check_parse_compatibility(text, [], split_bullets(text))
    assert "possible_multicolumn" in codes(outcome)


def test_short_document_is_critical():
    outcome = check_parse_compatibility("a@b.co too short", [], [])
    assert "too_little_text" in codes(outcome)


# --- content quality ------------------------------------------------------

def test_strong_bullets_score_clean():
    outcome = check_content_quality(split_bullets(GOOD), GOOD)
    assert "few_quantified_results" not in codes(outcome)
    assert "weak_bullet_openers" not in codes(outcome)


def test_unquantified_bullets_are_flagged():
    bullets = [
        "Worked on various backend services for the team",
        "Helped with the database and some other things",
        "Was involved in several projects across the company",
    ]
    assert "few_quantified_results" in codes(check_content_quality(bullets, " ".join(bullets)))


def test_passive_phrasing_is_flagged():
    text = "Responsible for the maintenance of the billing system and related duties"
    assert "passive_phrasing" in codes(check_content_quality([text], text))


def test_overlong_bullets_are_flagged():
    long_bullet = " ".join(["word"] * 60)
    assert "overlong_bullets" in codes(check_content_quality([long_bullet], long_bullet))


def test_no_bullets_at_all_is_flagged():
    assert "no_bullets" in codes(check_content_quality([], "some prose"))


# --- the arithmetic -------------------------------------------------------

def test_deductions_are_additive_and_inspectable():
    outcome = check_parse_compatibility("a@b.co short", ["no readable text"], [])
    assert outcome.deduction() == sum(f.points for f in outcome.findings)
    assert outcome.deduction() > 0


def test_every_finding_carries_an_actionable_message():
    outcome = check_parse_compatibility("short", ["no readable text"], [])
    for finding in outcome.findings:
        assert len(finding.message) > 40, finding.code
        assert finding.points > 0
        assert finding.severity in ("critical", "warning", "suggestion")
