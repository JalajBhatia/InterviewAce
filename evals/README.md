# Evaluation Harness

Runs simulated interview sessions against the real agent and measures its
behaviour, so claims about how it performs point at numbers rather than
assertions.

```bash
uv run python -m evals.run_eval --sessions 8     # quick check
uv run python -m evals.run_eval --sessions 50    # full run
uv run python -m evals.run_eval --persona weak   # one persona only
```

Results are written to `evals/results/latest.json` and `latest.md`.

## Why this is not part of `pytest`

It makes real LLM calls. `uv run pytest` stays fast, offline and free (105 tests,
under a second) because every test there replaces the LLM client with a fake.
Evaluation costs quota and needs network, so it is a deliberate act.

Roughly 8 requests per session, so a 50-session run is ~400 requests. `--pause`
controls the gap between calls to stay under per-minute limits.

## Candidates are scripted, on purpose

`personas.py` defines four candidates — **strong**, **weak**, **rambling**,
**terse** — as fixed answer banks with behavioural rules.

An LLM role-playing each persona would produce more realistic transcripts, but a
different answer every run, which makes a change in the pass rate impossible to
attribute: you could not tell a regression from noise. Deterministic input is
what makes the numbers comparable across runs.

The tradeoff being accepted: a scripted answer does not genuinely respond to the
question asked. That is acceptable because what is being measured is how the
interviewer reacts to an answer *of a given shape* — vague, terse, rambling,
detailed — not whether the answer was topical.

## What is measured, and how much to trust it

| Check | What it asserts | Kind |
|---|---|---|
| `one_question_at_a_time` | A turn contains at most one question-terminated sentence | deterministic |
| `probes_vague_answer` | Given a vague answer, the turn does **not** advance the question counter | deterministic |
| `understands_strong_answer` | Given a detailed answer, the turn is not a `rephrase` | deterministic |
| `feedback_has_verdict` | Verdict is one of the four allowed values | deterministic |
| `feedback_sections_populated` | Strengths / improvements / recommendations are non-empty | deterministic |
| `feedback_explains_verdict` | The verdict carries a real justification | deterministic |
| `role_coverage` | The session as a whole touches the role's subject matter | **heuristic** |
| `no_role_drift` | It does not use another role's vocabulary more than its own | **heuristic** |
| `feedback_grounded_in_transcript` | The assessment shares vocabulary with what was said | **heuristic** |

The heuristics are keyword overlap against the hand-written taxonomy in
`backend/roles.py`. They catch gross failures and are not proof. Every result
records which kind it is, and the report labels them, because a proxy presented
as a proof is worse than no measurement.

`probes_vague_answer` is the load-bearing check. It is only possible because the
interviewer declares what each turn did (`advance` / `follow_up` / `rephrase` /
`redirect`) through the `InterviewTurn` schema. Without that, "did it probe or
move on?" would need a human reader or an LLM judge — neither of which is
repeatable.

## Two things the harness found about itself

Worth recording, because both were cases of measuring the wrong thing:

1. **`stays_in_role` scored 0%** on the first run. It checked each question for
   role vocabulary, but *"describe the most challenging technical problem you
   faced"* is a perfectly good Backend Developer question containing none. It was
   measuring phrasing, not role fidelity. Now measured across the whole session,
   and split into coverage and drift.

2. **Advancing was treated as required** on a strong answer, so a legitimate
   follow-up counted as a failure — despite the system prompt explicitly telling
   the agent to ask one follow-up for depth. The check now only fails on
   `rephrase`, which genuinely means it failed to parse a clear answer.

## Interpreting a run

Results are not expected to be 100%. A harness that always passes is not
measuring anything. Sessions that error (rate limits, network) are counted
separately and excluded from behaviour rates, so a flaky run does not look like
the agent regressing.
