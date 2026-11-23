"""
Lexical matching between a resume and a target role or job description.

Why lexical, when embeddings exist
-----------------------------------
Because the thing being modelled *is* lexical. Recruiter search in an applicant
tracking system is exact-term: search "Kubernetes" and a resume saying "container
orchestration" does not appear in the results - not ranked lower, absent.

So term matching is not a cheap approximation of semantic similarity here. It is
a faithful model of the failure being warned about. A semantic matcher would score
"container orchestration" against "Kubernetes" as a strong match and tell the
candidate they are covered, which is confidently wrong advice that costs them the
interview. Its literalism is the feature.

The alias table in roles.py handles the cases worth handling - K8s/Kubernetes,
ReactJS/React - deterministically and inspectably.

BM25 rather than raw term frequency
------------------------------------
BM25 is the ranking function behind Lucene and Elasticsearch. Two properties
matter here:

- **Saturation**: the ninth mention of "Python" adds almost nothing over the
  third. Raw counting would let a resume win by repeating one keyword.
- **Length normalisation**: a three-page resume does not outscore a one-page one
  simply by containing more words.

Implemented here rather than imported. It is thirty lines, it removes a
dependency, and the parameters are choices worth being able to defend.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .roles import ROLE_SKILLS, SKILL_ALIASES, canonical_skill

# Standard BM25 parameters.
#   k1 controls how fast term frequency saturates. 1.5 is the usual default;
#      lower means the second mention of a term matters even less.
#   b  controls how much document length is normalised out. 0.75 is the usual
#      default: mostly normalised, not entirely.
# Left at the defaults deliberately - tuning them without labelled relevance data
# would be fitting to nothing.
K1 = 1.5
B = 0.75

_TOKEN = re.compile(r"[a-z0-9][a-z0-9+/.#-]*")
# Punctuation that is meaningful inside a skill name (node.js, ci/cd, c++) but
# not at the end of one, where it is just sentence punctuation.
_TRAILING = ".-/#+"


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens, keeping punctuation that lives inside skill names."""
    return [
        stripped
        for token in _TOKEN.findall(text.lower())
        if (stripped := token.rstrip(_TRAILING))
    ]


def normalise_terms(tokens: list[str]) -> list[str]:
    """
    Fold surface forms onto canonical skill names.

    This is what stops the matcher reporting a gap that does not exist: a resume
    saying "K8s" against a description asking for "Kubernetes" is a match, and a
    matcher that does not know that produces a wrong and actionable-looking
    finding.
    """
    return [canonical_skill(token) for token in tokens]


@dataclass
class TermScore:
    term: str
    score: float
    resume_count: int


@dataclass
class MatchResult:
    """
    Coverage of a target's vocabulary by a resume.

    ``score`` is a 0-1 coverage ratio, not a BM25 total: BM25 scores are unbounded
    and only meaningful relative to other documents in the same corpus, so
    presenting one as a percentage would be inventing precision. BM25 is used for
    what it is good at - *ranking* which matched terms matter most - while the
    headline number stays an interpretable "you cover N of M expected terms".
    """

    score: float
    matched: list[TermScore] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    target_description: str = ""

    @property
    def matched_terms(self) -> list[str]:
        return [term.term for term in self.matched]


def _idf(term: str, documents: list[list[str]]) -> float:
    """
    Inverse document frequency, BM25 variant.

    A term appearing in every document distinguishes nothing; a rare one is
    informative. The +0.5 smoothing and the outer max() keep the value positive
    for terms that appear in more than half the corpus, which the raw formula
    would otherwise send negative.
    """
    containing = sum(1 for document in documents if term in document)
    total = len(documents)
    return max(
        0.0,
        math.log((total - containing + 0.5) / (containing + 0.5) + 1.0),
    )


def bm25_scores(
    query_terms: list[str], document: list[str], corpus: list[list[str]]
) -> dict[str, float]:
    """
    Per-term BM25 contributions of ``document`` against ``query_terms``.

        score(t) = idf(t) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))

    where f is the term's frequency in the document. The denominator is where both
    properties come from: f appears on top and bottom so the ratio approaches
    (k1 + 1) as f grows - that is the saturation - and |D|/avgdl scales the
    penalty by how long this document is relative to the corpus average.
    """
    if not document or not corpus:
        return {}

    frequencies = Counter(document)
    doc_length = len(document)
    average_length = sum(len(d) for d in corpus) / len(corpus)

    scores: dict[str, float] = {}
    for term in set(query_terms):
        frequency = frequencies.get(term, 0)
        if not frequency:
            continue
        denominator = frequency + K1 * (1 - B + B * doc_length / average_length)
        scores[term] = _idf(term, corpus) * (frequency * (K1 + 1)) / denominator
    return scores


def _expand(skills: tuple[str, ...] | list[str]) -> list[str]:
    """A skill plus every surface form of it, so the corpus reflects real phrasing."""
    expanded: list[str] = []
    for skill in skills:
        expanded.extend(tokenize(skill))
        for alias in SKILL_ALIASES.get(skill, ()):
            expanded.extend(tokenize(alias))
    return expanded


def match_against_role(resume_text: str, role: str) -> MatchResult:
    """
    Fallback path: match against the curated skill list for a role.

    Weaker than matching a real job description, because it compares against an
    abstract job title rather than a specific requisition - but it always works,
    including when the user pastes nothing.
    """
    expected = ROLE_SKILLS.get(role, ())
    if not expected:
        return MatchResult(score=0.0, target_description=f"unknown role: {role}")

    document = normalise_terms(tokenize(resume_text))
    # Corpus = every role's vocabulary, so idf reflects which skills are common
    # across roles (python, sql) and which are distinctive (terraform, figma).
    corpus = [normalise_terms(_expand(skills)) for skills in ROLE_SKILLS.values()]

    present = set(document)
    matched_skills, missing_skills = [], []
    for skill in expected:
        forms = {canonical_skill(skill)} | {
            canonical_skill(a) for a in SKILL_ALIASES.get(skill, ())
        }
        tokens = {t for form in forms for t in normalise_terms(tokenize(form))}
        (matched_skills if tokens & present else missing_skills).append(skill)

    scores = bm25_scores(normalise_terms(_expand(expected)), document, corpus)
    frequencies = Counter(document)
    matched = sorted(
        (
            TermScore(
                skill,
                round(max(
                    (scores.get(t, 0.0) for t in normalise_terms(tokenize(skill))),
                    default=0.0,
                ), 4),
                max((frequencies.get(t, 0) for t in normalise_terms(tokenize(skill))), default=0),
            )
            for skill in matched_skills
        ),
        key=lambda term: term.score,
        reverse=True,
    )

    return MatchResult(
        score=round(len(matched_skills) / len(expected), 4),
        matched=matched,
        missing=missing_skills,
        target_description=f"{role} (curated skill list)",
    )


def match_against_job_description(resume_text: str, job_description: str) -> MatchResult:
    """
    Preferred path: match against a real posting.

    More accurate than the taxonomy, and closer to what an applicant tracking
    system actually does - it matches against a specific requisition, not a job
    title in the abstract.
    """
    resume_tokens = normalise_terms(tokenize(resume_text))
    jd_tokens = normalise_terms(tokenize(job_description))
    if not jd_tokens:
        return MatchResult(score=0.0, target_description="empty job description")

    # Terms worth reporting on. Very short tokens and generic filler carry no
    # signal, and idf would mostly discount them anyway - dropping them early
    # keeps the "missing terms" list readable.
    candidates = [t for t in set(jd_tokens) if len(t) > 2 and t not in STOPWORDS]

    # Canonical form -> what the job description actually wrote. Matching is done
    # canonically so "K8s" on the resume covers "Kubernetes" in the posting, but
    # gaps must be reported in the posting's own words: the whole reason for
    # choosing lexical matching is that recruiter search is exact-term, so telling
    # someone to add "infrastructure as code" when the posting says "Terraform"
    # would be precisely the wrong advice.
    surface_forms: dict[str, str] = {}
    for raw in tokenize(job_description):
        surface_forms.setdefault(canonical_skill(raw), raw)

    # Corpus is the JD plus every role's vocabulary, so idf has something to
    # measure rarity against rather than a single document.
    corpus = [jd_tokens] + [normalise_terms(_expand(s)) for s in ROLE_SKILLS.values()]
    scores = bm25_scores(candidates, resume_tokens, corpus)

    frequencies = Counter(resume_tokens)
    present = set(resume_tokens)
    matched = sorted(
        (TermScore(t, round(scores.get(t, 0.0), 4), frequencies[t]) for t in candidates if t in present),
        key=lambda term: term.score,
        reverse=True,
    )
    missing = sorted(surface_forms.get(t, t) for t in candidates if t not in present)

    return MatchResult(
        score=round(len(matched) / len(candidates), 4) if candidates else 0.0,
        matched=matched,
        missing=missing,
        target_description="pasted job description",
    )


STOPWORDS = frozenset("""
the and for with you your our are will able have has that this from they their
who whom what when which about into out over under more most such than then
work working works role roles team teams company companies year years experience
experiences strong good great excellent ability abilities including include
includes required require requires requirements preferred plus etc
""".split())
