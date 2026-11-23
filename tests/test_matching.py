"""BM25 term matching against a role taxonomy or a pasted job description."""

import pytest

from backend.matching import (
    B, K1, bm25_scores, match_against_job_description, match_against_role,
    normalise_terms, tokenize,
)

BACKEND_RESUME = """
Senior Backend Engineer. Built REST APIs in Python with FastAPI serving 2 million
requests a day. Migrated the datastore to PostgreSQL and added Redis caching.
Introduced Docker and CI/CD pipelines. Ran production K8s clusters.
Designed the authentication layer and led system design reviews.
"""

DESIGNER_RESUME = """
Product Designer. Ran user research and usability testing, built wireframes and
prototypes in Figma, and maintained the design system across three products.
"""


# --- the algorithm itself -------------------------------------------------

def test_term_frequency_saturates():
    """The point of BM25 over raw counting: the ninth mention adds almost nothing."""
    corpus = [["python"] * 3, ["java"] * 3, ["sql"] * 3]
    few = bm25_scores(["python"], ["python"] * 2 + ["other"] * 8, corpus)["python"]
    many = bm25_scores(["python"], ["python"] * 20 + ["other"] * 8, corpus)["python"]

    assert many > few
    assert many < few * 3, "10x the mentions must not give anywhere near 10x the score"


def test_longer_documents_are_normalised():
    """A padded resume must not outscore a concise one on the same evidence."""
    corpus = [["python", "sql"], ["java"], ["go"]]
    concise = bm25_scores(["python"], ["python", "sql"], corpus)["python"]
    padded = bm25_scores(["python"], ["python"] + ["filler"] * 200, corpus)["python"]
    assert concise > padded


def test_common_terms_score_lower_than_rare_ones():
    """idf: a term in every document distinguishes nothing."""
    corpus = [["python", "terraform"], ["python", "java"], ["python", "sql"]]
    scores = bm25_scores(["python", "terraform"], ["python", "terraform"], corpus)
    assert scores["terraform"] > scores["python"]


def test_idf_never_goes_negative():
    """The raw formula sends terms appearing in >half the corpus negative."""
    corpus = [["x"], ["x"], ["x"]]
    assert bm25_scores(["x"], ["x"], corpus)["x"] >= 0


def test_parameters_are_the_standard_defaults():
    assert (K1, B) == (1.5, 0.75)


def test_absent_terms_are_not_scored():
    assert "rust" not in bm25_scores(["rust"], ["python"], [["python"], ["java"]])


def test_empty_inputs_are_safe():
    assert bm25_scores(["x"], [], [["y"]]) == {}
    assert bm25_scores(["x"], ["y"], []) == {}


# --- tokenising and aliases -----------------------------------------------

def test_skill_punctuation_survives_tokenising():
    assert "ci/cd" in tokenize("we used ci/cd pipelines")
    assert "node.js" in tokenize("built in node.js")


def test_aliases_fold_onto_canonical_names():
    assert "kubernetes" in normalise_terms(tokenize("we ran k8s in production"))
    assert "react" in normalise_terms(tokenize("built the reactjs frontend"))


# --- matching against a role ----------------------------------------------

def test_backend_resume_covers_backend_skills():
    result = match_against_role(BACKEND_RESUME, "Backend Developer")
    assert result.score > 0.5
    assert "python" in result.matched_terms
    assert "postgresql" in result.matched_terms


def test_alias_only_mention_still_counts_as_coverage():
    """The resume says K8s; the taxonomy says kubernetes. Not a gap."""
    result = match_against_role(BACKEND_RESUME, "DevOps Engineer")
    assert "kubernetes" in result.matched_terms


def test_mismatched_resume_scores_low_and_lists_gaps():
    result = match_against_role(DESIGNER_RESUME, "Backend Developer")
    assert result.score < 0.3
    assert "postgresql" in result.missing


def test_matched_terms_are_ranked_by_informativeness():
    result = match_against_role(BACKEND_RESUME, "Backend Developer")
    scores = [term.score for term in result.matched]
    assert scores == sorted(scores, reverse=True)


def test_unknown_role_fails_safely():
    assert match_against_role(BACKEND_RESUME, "Astronaut").score == 0.0


def test_score_is_coverage_not_a_raw_bm25_total():
    """BM25 totals are unbounded; presenting one as a percentage invents precision."""
    result = match_against_role(BACKEND_RESUME, "Backend Developer")
    assert 0.0 <= result.score <= 1.0


# --- matching against a job description -----------------------------------

JD = """
We are looking for a Backend Engineer with strong Python experience.
You will design REST APIs, work with PostgreSQL, and deploy services on
Kubernetes. Experience with Terraform and observability tooling is preferred.
"""


def test_job_description_matching_finds_covered_and_missing_terms():
    result = match_against_job_description(BACKEND_RESUME, JD)
    assert "python" in result.matched_terms
    assert "postgresql" in result.matched_terms
    assert "terraform" in result.missing


def test_gaps_are_reported_in_the_posting_s_own_words():
    """
    The posting says "Terraform"; the taxonomy calls it "infrastructure as code".
    Telling the candidate to add the canonical name would be the wrong advice,
    because recruiter search is exact-term - which is why lexical matching was
    chosen in the first place.
    """
    result = match_against_job_description(BACKEND_RESUME, JD)
    assert "terraform" in result.missing
    assert "infrastructure as code" not in result.missing


def test_tokens_do_not_keep_sentence_punctuation():
    assert "kubernetes" in tokenize("deploy on Kubernetes.")
    assert "kubernetes." not in tokenize("deploy on Kubernetes.")
    assert "node.js" in tokenize("built in node.js"), "internal dots must survive"


def test_stopwords_are_excluded_from_the_gap_list():
    result = match_against_job_description(BACKEND_RESUME, JD)
    for filler in ("the", "with", "you", "will", "experience"):
        assert filler not in result.missing


def test_empty_job_description_is_safe():
    assert match_against_job_description(BACKEND_RESUME, "").score == 0.0


def test_target_is_named_so_the_user_knows_what_was_compared():
    assert "job description" in match_against_job_description(BACKEND_RESUME, JD).target_description
    assert "curated" in match_against_role(BACKEND_RESUME, "Backend Developer").target_description
