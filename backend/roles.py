"""
Role taxonomy: what each supported role is about, and what its skills look like
in the wild.

Data, not logic, on purpose. Three separate things consume it:

1. Role inference - deterministic cross-check on the model's guess.
2. Resume/JD matching - the fallback target when no job description is pasted.
3. Alias normalisation before term matching - "K8s" and "Kubernetes" are the same
   skill, and a lexical matcher that does not know that will report a gap that
   is not there.

That third use is why the aliases exist. It handles most of what a semantic model
would catch, while staying inspectable: someone can read this file and tell you
it is wrong, which is not true of an embedding space.
"""

from __future__ import annotations

# Canonical skill -> surface forms that mean the same thing. Lower-case; matching
# is case-insensitive. The canonical name is itself an implicit alias.
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "python": ("py",),
    "javascript": ("js", "ecmascript"),
    "typescript": ("ts",),
    "react": ("reactjs", "react.js"),
    "node.js": ("node", "nodejs"),
    "kubernetes": ("k8s",),
    "docker": ("containerisation", "containerization", "containers"),
    "ci/cd": ("cicd", "continuous integration", "continuous delivery", "continuous deployment"),
    "machine learning": ("ml",),
    "deep learning": ("dl", "neural networks"),
    "natural language processing": ("nlp",),
    "postgresql": ("postgres", "psql"),
    "mongodb": ("mongo",),
    "amazon web services": ("aws",),
    "google cloud platform": ("gcp", "google cloud"),
    "microsoft azure": ("azure",),
    "rest api": ("rest", "restful", "restful api", "rest apis"),
    "graphql": ("graph ql",),
    "sql": ("structured query language",),
    "a/b testing": ("ab testing", "split testing"),
    "user experience": ("ux",),
    "user interface": ("ui",),
    "search engine optimization": ("seo",),
    "key performance indicator": ("kpi", "kpis"),
    "customer relationship management": ("crm",),
    "infrastructure as code": ("iac", "terraform"),
    "observability": ("monitoring", "telemetry"),
    "object oriented programming": ("oop",),
    "test driven development": ("tdd",),
    "behavioral questions": ("behavioral", "behavioural", "star method", "hr questions"),
    "problem solving": ("problem-solving", "analytical skills"),
    "culture fit": ("cultural fit", "company culture"),
    "conflict resolution": ("handling conflict", "resolving disagreements"),
    "teamwork": ("collaboration", "cross-functional collaboration"),
}


# Role -> the skills an interviewer would expect to see. Deliberately short:
# these are signals for matching and inference, not exhaustive job descriptions.
ROLE_SKILLS: dict[str, tuple[str, ...]] = {
    "Software Development Intern": (
        "python", "java", "algorithms", "data structures", "git",
        "object oriented programming", "debugging", "problem solving",
    ),
    "Frontend Developer": (
        "javascript", "typescript", "react", "html", "css", "accessibility",
        "responsive design", "web performance", "user interface",
    ),
    "Backend Developer": (
        "python", "java", "rest api", "sql", "postgresql", "redis", "caching",
        "authentication", "system design", "microservices", "concurrency",
    ),
    "Data Analyst": (
        "sql", "excel", "tableau", "power bi", "data visualization",
        "key performance indicator", "reporting", "python", "statistics",
    ),
    "Machine Learning Intern": (
        "python", "machine learning", "deep learning", "pandas", "numpy",
        "scikit-learn", "pytorch", "tensorflow", "data visualization",
    ),
    "HR Interview": (
        "behavioral questions", "communication", "teamwork", "conflict resolution",
        "leadership", "culture fit", "recruiting", "employee relations",
    ),
    "Software Engineer": (
        "python", "java", "algorithms", "data structures", "system design",
        "git", "object oriented programming", "test driven development", "debugging",
    ),
    "Frontend Developer": (
        "javascript", "typescript", "react", "html", "css", "accessibility",
        "responsive design", "web performance", "user interface",
    ),
    "Backend Developer": (
        "python", "java", "rest api", "sql", "postgresql", "redis", "caching",
        "authentication", "system design", "microservices", "concurrency",
    ),
    "Full Stack Developer": (
        "javascript", "react", "node.js", "rest api", "sql", "html", "css",
        "docker", "system design",
    ),
    "DevOps Engineer": (
        "docker", "kubernetes", "ci/cd", "infrastructure as code", "linux",
        "amazon web services", "observability", "bash", "reliability",
    ),
    "Data Scientist": (
        "python", "statistics", "machine learning", "pandas", "numpy",
        "scikit-learn", "experimentation", "a/b testing", "data visualization",
    ),
    "Data Analyst": (
        "sql", "excel", "tableau", "power bi", "data visualization",
        "key performance indicator", "reporting", "python", "statistics",
    ),
    "Machine Learning Engineer": (
        "python", "machine learning", "deep learning", "pytorch", "tensorflow",
        "mlops", "model deployment", "feature engineering", "docker",
    ),
    "Product Manager": (
        "product strategy", "roadmap", "prioritization", "user research",
        "key performance indicator", "a/b testing", "stakeholder management",
        "market analysis",
    ),
    "Project Manager": (
        "project planning", "agile", "scrum", "risk management", "stakeholder management",
        "budgeting", "jira", "delivery",
    ),
    "Business Analyst": (
        "requirements gathering", "process mapping", "sql", "stakeholder management",
        "key performance indicator", "business case", "documentation",
    ),
    "UX/UI Designer": (
        "user experience", "user interface", "figma", "wireframing", "prototyping",
        "user research", "usability testing", "accessibility", "design systems",
    ),
    "Cybersecurity Analyst": (
        "threat detection", "incident response", "vulnerability assessment",
        "penetration testing", "siem", "network security", "encryption", "compliance",
    ),
    "Sales Associate": (
        "lead qualification", "objection handling", "closing", "pipeline management",
        "customer relationship management", "negotiation", "prospecting",
    ),
    "Marketing Manager": (
        "campaign management", "search engine optimization", "content strategy",
        "brand positioning", "marketing analytics", "a/b testing", "segmentation",
    ),
    "Customer Support": (
        "ticketing", "conflict resolution", "empathy", "troubleshooting",
        "customer relationship management", "escalation", "documentation",
    ),
    "HR Manager": (
        "recruiting", "performance management", "employee relations", "onboarding",
        "hr policy", "compensation", "conflict resolution",
    ),
    "Financial Analyst": (
        "financial modeling", "valuation", "forecasting", "excel",
        "financial statements", "variance analysis", "budgeting",
    ),
}

# The exact strings the frontend dropdown offers. Inference is constrained to
# these, so a detected role is always something the rest of the app understands.
SUPPORTED_ROLES: tuple[str, ...] = tuple(ROLE_SKILLS)


def _build_alias_lookup() -> dict[str, str]:
    """Flatten SKILL_ALIASES into surface form -> canonical skill."""
    lookup: dict[str, str] = {}
    for canonical, aliases in SKILL_ALIASES.items():
        lookup[canonical] = canonical
        for alias in aliases:
            lookup[alias] = canonical
    return lookup


ALIAS_TO_CANONICAL = _build_alias_lookup()


def canonical_skill(term: str) -> str:
    """
    Map a surface form to its canonical skill name.

    Unknown terms pass through unchanged rather than being dropped: the taxonomy
    is a normalisation aid, not an allow-list, and a skill we have not catalogued
    is still a skill the candidate has.
    """
    return ALIAS_TO_CANONICAL.get(term.strip().lower(), term.strip().lower())
