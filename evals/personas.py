"""
Scripted candidate personas.

Deterministic on purpose. The harness exists to produce numbers that can be
compared across runs, so a change in the pass rate means the agent changed. An
LLM-simulated candidate would give more realistic transcripts but a different
answer every run, which would make a regression indistinguishable from noise.

The tradeoff being accepted: a scripted answer does not actually respond to the
question that was asked. That is fine for what is being measured - whether the
interviewer follows up on a vague answer, redirects a rambler, and asks one
question at a time are all properties of *its* behaviour given an answer of a
certain shape, not of whether the answer was topical.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    """
    A scripted candidate.

    ``expects_followup`` is the assertion this persona exists to test: for weak
    and terse candidates the interviewer should probe rather than move on, so a
    turn that advances the question counter is a failure. For strong candidates
    the opposite - endlessly probing a complete answer wastes the interview.
    """

    name: str
    description: str
    answers: tuple[str, ...]
    expects_followup: bool
    # None means "no expectation" - used where either behaviour is defensible.
    expects_advance: bool | None = None
    filler: str = "I think that covers it."

    def answer(self, turn_index: int) -> str:
        """Answers are consumed in order, then the filler repeats."""
        if turn_index < len(self.answers):
            return self.answers[turn_index]
        return self.filler


STRONG = Persona(
    name="strong",
    description="Specific, quantified answers with concrete examples.",
    expects_followup=False,
    expects_advance=True,
    filler=(
        "I led that work end to end. We measured it with a dashboard tracking p95 "
        "latency and error rate, and I wrote the runbook the on-call team still uses."
    ),
    answers=(
        "I'm a backend engineer with four years' experience. Most recently I owned "
        "the payments service at a fintech - a FastAPI application handling about "
        "2 million requests a day, backed by PostgreSQL with Redis for hot reads.",
        "The hardest problem was a latency regression after we sharded the database. "
        "p95 went from 90ms to 600ms. I profiled it and found we'd lost an index on "
        "the shard key, so every lookup was a full scan across shards. Adding a "
        "composite index on (tenant_id, created_at) brought it back to 85ms.",
        "I'd use a read-through cache with a short TTL, around 30 seconds, and "
        "invalidate on write. The tradeoff is a brief window of stale data, which "
        "was acceptable for balance display but not for the ledger itself - that "
        "reads from the primary every time.",
        "We ran it as a blue-green deploy behind a feature flag, rolled out to 5% "
        "of traffic for an hour, watched error rates, then went to 100%. Rollback "
        "was a flag flip, so under thirty seconds.",
    ),
)

WEAK = Persona(
    name="weak",
    description="Vague, uncertain, no specifics. Should be probed, not advanced past.",
    expects_followup=True,
    expects_advance=False,
    filler="I'm not really sure, sorry.",
    answers=(
        "Um, I've done a bit of backend stuff I guess.",
        "I don't really know. We used a database.",
        "Not sure what you mean by that.",
        "I think someone else handled that part.",
    ),
)

RAMBLING = Persona(
    name="rambling",
    description="Very long, drifting off-topic. Should be refocused.",
    expects_followup=False,
    filler=(
        "Anyway, that reminds me of another time - so we were at this offsite, and "
        "the whole team had opinions about the coffee machine, which honestly says a "
        "lot about engineering culture in general, and by the way I've been meaning "
        "to mention that my old manager had this theory about standups, which is "
        "that they're really just a ritual, and speaking of rituals, the deployment "
        "process at that company was almost ceremonial, we had a whole checklist, "
        "and someone printed it out and laminated it, which I found hilarious."
    ),
    answers=(
        "Oh wow, okay, so where do I start. I studied physics originally, which is "
        "a funny story because I only picked it because my brother did, and he's now "
        "a dentist, which tells you something. Anyway I got into programming through "
        "modding games, and by the way I still play a lot, which my partner finds "
        "annoying, and speaking of which we just moved house, which was chaos, and "
        "the movers broke my monitor, so I've been working on a laptop, which is "
        "relevant because ergonomics really matter for developers, and I read this "
        "article about standing desks that I could send you.",
        "Right, so that project - although actually before that I should explain the "
        "team structure, because it was unusual, we had two tech leads which never "
        "works, and funny story, one of them left mid-project to start a bakery. A "
        "bakery! Anyway the codebase was Python, though parts were Ruby for historical "
        "reasons nobody could explain, and that reminds me of a talk I saw about "
        "legacy code, the speaker had this great analogy about archaeology.",
    ),
)

TERSE = Persona(
    name="terse",
    description="Correct but extremely brief. Should be asked to elaborate.",
    expects_followup=True,
    expects_advance=False,
    filler="Yes.",
    answers=("Backend dev.", "Postgres.", "Caching.", "Docker.", "Four years."),
)

ALL_PERSONAS: tuple[Persona, ...] = (STRONG, WEAK, RAMBLING, TERSE)

PERSONAS_BY_NAME = {persona.name: persona for persona in ALL_PERSONAS}
