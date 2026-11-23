"""
Deterministic resume checks.

Two families, both rule-based and both fully reproducible:

- **Parse compatibility** - will an automated reader actually see this document?
  These are the most objectively defensible findings in the whole feature. A
  resume whose text cannot be extracted is broken as a fact, not as a matter of
  taste, and every finding maps to a concrete fix.

- **Content quality** - conventions recruiters expect. Weakest of the signals,
  deliberately weighted lowest: a bullet without a number is a style preference,
  whereas an unreadable PDF is a defect.

Nothing here calls an LLM. The score has to be reproducible to be assertable in
the eval harness and explainable to the user, so every point deducted traces to a
named rule. The LLM's job comes later and is limited to phrasing these findings
readably - it never invents one and never moves the number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["critical", "warning", "suggestion"]

# Headings automated readers look for when splitting a resume into sections.
# A resume titled "My Journey" instead of "Experience" defeats that split.
EXPECTED_SECTIONS: dict[str, tuple[str, ...]] = {
    "experience": ("experience", "employment", "work history", "professional background"),
    "education": ("education", "academic", "qualifications"),
    "skills": ("skills", "technical skills", "technologies", "competencies"),
}

ACTION_VERBS = frozenset("""
built designed developed led managed implemented created launched shipped
migrated optimised optimized reduced increased improved automated architected
delivered owned drove scaled refactored integrated deployed established
mentored coordinated analysed analyzed negotiated resolved streamlined
""".split())

# A date range in any of the forms resumes commonly use.
DATE_RANGE = re.compile(
    r"(19|20)\d{2}\s*[-–—]\s*((19|20)\d{2}|present|current|now)"
    r"|[a-z]{3,9}\s+(19|20)\d{2}\s*[-–—]\s*([a-z]{3,9}\s+(19|20)\d{2}|present|current)",
    re.IGNORECASE,
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
URL = re.compile(r"(linkedin\.com/\S+|github\.com/\S+|https?://\S+)", re.IGNORECASE)
QUANTIFIED = re.compile(r"(\d+(\.\d+)?\s*(%|percent|x\b)|[£$€]\s?\d|\b\d{2,}\b)")


@dataclass
class Finding:
    """
    One thing wrong, with the cost attached.

    ``points`` is what this deducts from its category. Findings carry their own
    weight so the arithmetic is inspectable rather than buried in a scorer.
    """

    code: str
    severity: Severity
    message: str
    points: int
    evidence: str = ""


@dataclass
class CheckOutcome:
    findings: list[Finding] = field(default_factory=list)

    def deduction(self) -> int:
        return sum(finding.points for finding in self.findings)


def check_parse_compatibility(
    text: str, parse_warnings: list[str], bullets: list[str]
) -> CheckOutcome:
    """
    Will an automated reader see this document at all?

    This is what "ATS-friendly" actually means. Real applicant tracking systems
    are databases with keyword search over parsed text - if the parse fails, the
    contents are invisible no matter how good they are.
    """
    outcome = CheckOutcome()
    lowered = text.lower()

    # The parser already noticed things while extracting. Those are the highest
    # confidence findings available: they are observations, not inferences.
    for warning in parse_warnings:
        if "no readable text" in warning or "image" in warning.lower():
            outcome.findings.append(
                Finding(
                    "no_text_layer", "critical",
                    "Part or all of this document has no text layer, so an automated "
                    "reader sees nothing there. Export from the original editor as PDF "
                    "rather than scanning or exporting an image.",
                    30, warning,
                )
            )
        elif "table" in warning.lower():
            outcome.findings.append(
                Finding(
                    "uses_tables", "warning",
                    "Tables are read unreliably - contents are often flattened into the "
                    "wrong order or dropped. Use a single-column layout without tables.",
                    8, warning,
                )
            )

    if len(text.strip()) < 400:
        outcome.findings.append(
            Finding(
                "too_little_text", "critical",
                f"Only {len(text.strip())} characters were extracted. A complete resume "
                "normally yields well over a thousand.",
                20,
            )
        )

    # Contact details are the first thing a parser tries to pull out, and they are
    # frequently placed in a header, which many parsers skip entirely.
    if not EMAIL.search(text):
        outcome.findings.append(
            Finding(
                "no_email", "critical",
                "No email address found in the document body. If yours is in a page "
                "header, move it into the body - many parsers skip header regions.",
                12,
            )
        )
    if not PHONE.search(text) and not URL.search(text):
        outcome.findings.append(
            Finding(
                "no_contact_channel", "warning",
                "No phone number or profile link found. Include at least one besides email.",
                5,
            )
        )

    missing = [
        name for name, aliases in EXPECTED_SECTIONS.items()
        if not any(alias in lowered for alias in aliases)
    ]
    if missing:
        outcome.findings.append(
            Finding(
                "nonstandard_sections", "warning",
                "No recognisable heading for: " + ", ".join(sorted(missing))
                + ". Automated readers split resumes on standard headings, so creative "
                "titles like \"My Journey\" can cause whole sections to be missed.",
                6 * len(missing),
                f"missing: {sorted(missing)}",
            )
        )

    if not DATE_RANGE.search(text):
        outcome.findings.append(
            Finding(
                "no_parseable_dates", "warning",
                "No date ranges in a recognisable format. Use \"Jan 2022 - Present\" or "
                "\"2022 - 2024\" so employment history can be reconstructed.",
                8,
            )
        )

    # A very low bullet count against a document with plenty of text usually means
    # the text extracted as one run - the classic symptom of a multi-column layout
    # being read across the columns instead of down them.
    if len(text) > 1500 and len(bullets) < 5:
        outcome.findings.append(
            Finding(
                "possible_multicolumn", "warning",
                "Text extracted as very few distinct lines for its length, which "
                "usually means a multi-column layout being read across columns rather "
                "than down them. A single-column layout extracts predictably.",
                10,
                f"{len(bullets)} lines from {len(text)} characters",
            )
        )

    return outcome


def check_content_quality(bullets: list[str], text: str) -> CheckOutcome:
    """
    Conventions recruiters expect.

    Weighted lowest on purpose. These are style, and a resume can be excellent
    while breaking them - which is exactly why a failure here must never outweigh
    a document that cannot be parsed.
    """
    outcome = CheckOutcome()

    if not bullets:
        outcome.findings.append(
            Finding("no_bullets", "warning",
                    "No bullet-style achievement lines found. Most resumes describe "
                    "experience as bullets rather than paragraphs.", 8)
        )
        return outcome

    quantified = [b for b in bullets if QUANTIFIED.search(b)]
    ratio = len(quantified) / len(bullets)
    if ratio < 0.3:
        outcome.findings.append(
            Finding(
                "few_quantified_results", "suggestion",
                f"Only {len(quantified)} of {len(bullets)} bullets contain a number. "
                "Quantified outcomes (\"cut p95 latency from 400ms to 90ms\") are far "
                "more credible than unmeasured claims.",
                8 if ratio < 0.15 else 4,
                f"{ratio:.0%} quantified",
            )
        )

    starts_with_verb = sum(
        1 for b in bullets if b.split() and b.split()[0].lower().strip(",.") in ACTION_VERBS
    )
    if starts_with_verb / len(bullets) < 0.4:
        outcome.findings.append(
            Finding(
                "weak_bullet_openers", "suggestion",
                f"Only {starts_with_verb} of {len(bullets)} bullets open with an action "
                "verb. Starting with what you did (\"Built\", \"Led\", \"Reduced\") reads "
                "more decisively than \"Responsible for...\".",
                5,
                f"{starts_with_verb}/{len(bullets)}",
            )
        )

    overlong = [b for b in bullets if len(b.split()) > 45]
    if overlong:
        outcome.findings.append(
            Finding(
                "overlong_bullets", "suggestion",
                f"{len(overlong)} bullet(s) run past 45 words. Long bullets get skimmed; "
                "split them or cut to the outcome.",
                3,
                overlong[0][:80],
            )
        )

    if re.search(r"\bresponsible for\b|\bduties included\b|\btasked with\b", text, re.I):
        outcome.findings.append(
            Finding(
                "passive_phrasing", "suggestion",
                "Phrases like \"responsible for\" describe a job description rather than "
                "an achievement. Say what you actually did and what changed.",
                3,
            )
        )

    return outcome
