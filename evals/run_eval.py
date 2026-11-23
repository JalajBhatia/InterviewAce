"""
Run simulated interview sessions and measure the agent's behaviour.

This is NOT part of the pytest suite. It makes real LLM calls, so it costs quota
and needs network, and it is run deliberately:

    uv run python -m evals.run_eval --sessions 8      # quick check
    uv run python -m evals.run_eval --sessions 50     # the real run

Results are written to evals/results/ so the numbers can be cited rather than
claimed.

What "failure" means here
-------------------------
A failed check means the agent behaved differently from the expectation, not
that the harness broke. Transport errors are counted separately and excluded
from behaviour rates - otherwise a flaky network would show up as the interview
agent getting worse.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backend.config import BASE_DIR
from backend.graph import interview_graph, thread_config
from backend.llm_client import LLMUnavailableError, MODEL_NAME
from backend.main import (
    ChatRequest, FeedbackRequest, StartSessionRequest,
    chat, get_feedback, start_session, transcript_from_checkpoint,
)
from evals.checks import (
    check_feedback_well_formed, check_one_question_at_a_time,
    check_stays_in_role, check_turn_action,
)
from evals.personas import ALL_PERSONAS, Persona

RESULTS_DIR = BASE_DIR / "evals" / "results"

# Roles exercised. Deliberately spread across the taxonomy rather than all
# technical, so "stays in role" is measured somewhere it could plausibly fail.
EVAL_ROLES = (
    ("Backend Developer", "Senior"),
    ("Data Scientist", "Mid-Level"),
    ("Product Manager", "Mid-Level"),
    ("DevOps Engineer", "Senior"),
    ("UX/UI Designer", "Junior"),
)

# Independent of the question budget: a persona that always triggers a rephrase
# would otherwise never terminate.
MAX_TURNS = 6


@dataclass
class SessionResult:
    session_id: str
    persona: str
    role: str
    experience_level: str
    turns: int
    questions_asked: int
    actions: list[str] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    verdict: str | None = None
    duration_seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_session(persona: Persona, role: str, experience_level: str, pause: float) -> SessionResult:
    """Drive one full interview with a scripted candidate, checking every turn."""
    started = time.perf_counter()
    result = SessionResult(
        session_id="", persona=persona.name, role=role,
        experience_level=experience_level, turns=0, questions_asked=0,
    )

    try:
        opening = start_session(
            StartSessionRequest(role=role, experience_level=experience_level)
        )
        session_id = opening["session_id"]
        result.session_id = session_id

        # The greeting is a turn the candidate sees, so it is checked too.
        result.checks.append(asdict(check_one_question_at_a_time(opening["initial_message"])))
        questions_asked_text = [opening["initial_message"]]

        for turn_index in range(MAX_TURNS):
            time.sleep(pause)
            reply = chat(
                ChatRequest(session_id=session_id, user_message=persona.answer(turn_index))
            )
            result.turns += 1

            state = interview_graph.get_state(thread_config(session_id)).values
            action = state.get("last_action", "")
            result.actions.append(action)

            if reply.is_interview_over:
                break

            questions_asked_text.append(reply.agent_message)
            result.checks.append(asdict(check_one_question_at_a_time(reply.agent_message)))
            result.checks.append(asdict(check_turn_action(action, persona.expects_followup)))

        # Role fidelity is a property of the interview, not of any single turn.
        result.checks.extend(
            asdict(check) for check in check_stays_in_role(questions_asked_text, role)
        )

        state = interview_graph.get_state(thread_config(session_id)).values
        result.questions_asked = state.get("questions_asked", 0)

        time.sleep(pause)
        feedback = get_feedback(FeedbackRequest(session_id=session_id)).feedback
        result.verdict = feedback.verdict

        transcript = " ".join(
            turn["content"] for turn in transcript_from_checkpoint(session_id)
        )
        result.checks.extend(
            asdict(check) for check in check_feedback_well_formed(feedback, transcript)
        )

    except LLMUnavailableError as exc:
        # Transport, not behaviour. Recorded separately so a rate-limited run
        # does not look like the agent regressing.
        result.error = f"llm_unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001 - a harness must not die mid-run
        result.error = f"{type(exc).__name__}: {exc}"

    result.duration_seconds = round(time.perf_counter() - started, 2)
    return result


def summarise(results: list[SessionResult]) -> dict:
    """Aggregate into the numbers that get cited."""
    completed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    by_check: dict[str, Counter] = defaultdict(Counter)
    deterministic: dict[str, bool] = {}
    for result in completed:
        for check in result.checks:
            by_check[check["name"]]["pass" if check["passed"] else "fail"] += 1
            deterministic[check["name"]] = check["deterministic"]

    checks_summary = {}
    for name, counts in sorted(by_check.items()):
        total = counts["pass"] + counts["fail"]
        checks_summary[name] = {
            "passed": counts["pass"],
            "total": total,
            "rate": round(counts["pass"] / total, 4) if total else 0.0,
            "deterministic": deterministic[name],
        }

    per_persona = {}
    for persona in ALL_PERSONAS:
        subset = [r for r in completed if r.persona == persona.name]
        if not subset:
            continue
        passed = sum(1 for r in subset for c in r.checks if c["passed"])
        total = sum(len(r.checks) for r in subset)
        per_persona[persona.name] = {
            "sessions": len(subset),
            "check_pass_rate": round(passed / total, 4) if total else 0.0,
            "avg_questions_asked": round(
                statistics.mean(r.questions_asked for r in subset), 2
            ),
            "actions": dict(Counter(a for r in subset for a in r.actions)),
            "verdicts": dict(Counter(r.verdict for r in subset if r.verdict)),
        }

    all_checks = [c for r in completed for c in r.checks]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "sessions_run": len(results),
        "sessions_completed": len(completed),
        "sessions_errored": len(failed),
        "errors": [r.error for r in failed],
        "total_checks": len(all_checks),
        "checks_passed": sum(1 for c in all_checks if c["passed"]),
        "overall_pass_rate": round(
            sum(1 for c in all_checks if c["passed"]) / len(all_checks), 4
        ) if all_checks else 0.0,
        "avg_session_seconds": round(
            statistics.mean(r.duration_seconds for r in completed), 2
        ) if completed else 0.0,
        "by_check": checks_summary,
        "by_persona": per_persona,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Evaluation Results",
        "",
        f"- **Generated:** {summary['generated_at']}",
        f"- **Model:** `{summary['model']}`",
        f"- **Sessions:** {summary['sessions_completed']} completed"
        f" ({summary['sessions_errored']} errored)",
        f"- **Checks:** {summary['checks_passed']}/{summary['total_checks']}"
        f" passed ({summary['overall_pass_rate']:.1%})",
        f"- **Avg session:** {summary['avg_session_seconds']}s",
        "",
        "## By check",
        "",
        "| Check | Passed | Rate | Kind |",
        "|---|---|---|---|",
    ]
    for name, stats in summary["by_check"].items():
        kind = "deterministic" if stats["deterministic"] else "heuristic"
        lines.append(
            f"| `{name}` | {stats['passed']}/{stats['total']} | {stats['rate']:.1%} | {kind} |"
        )

    lines += ["", "## By persona", "", "| Persona | Sessions | Pass rate | Avg questions | Actions |", "|---|---|---|---|---|"]
    for name, stats in summary["by_persona"].items():
        actions = ", ".join(f"{k}×{v}" for k, v in sorted(stats["actions"].items()) if k)
        lines.append(
            f"| {name} | {stats['sessions']} | {stats['check_pass_rate']:.1%} | "
            f"{stats['avg_questions_asked']} | {actions} |"
        )

    lines += [
        "",
        "## Reading these numbers",
        "",
        "`stays_in_role` and `feedback_grounded_in_transcript` are **heuristics**, not",
        "proofs. The first looks for role vocabulary in the question and the second for",
        "shared terms between transcript and assessment. Both catch gross failures and",
        "neither should be read as a guarantee. Every other check is a deterministic",
        "property of the transcript.",
        "",
        "`follows_up_on_weak_answer` is the load-bearing one: the interviewer declares",
        "what each turn did (`advance` / `follow_up` / `rephrase` / `redirect`), so",
        "whether it probed a vague answer instead of moving on is directly observable",
        "rather than inferred.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run simulated interview evaluations.")
    parser.add_argument("--sessions", type=int, default=8,
                        help="Number of sessions to run (default 8).")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds between LLM calls, to stay under per-minute limits.")
    parser.add_argument("--persona", choices=[p.name for p in ALL_PERSONAS],
                        help="Restrict to a single persona.")
    parser.add_argument("--out", default=str(RESULTS_DIR),
                        help="Directory for results.")
    args = parser.parse_args()

    personas = (
        [p for p in ALL_PERSONAS if p.name == args.persona] if args.persona else list(ALL_PERSONAS)
    )

    plan = [
        (personas[i % len(personas)], *EVAL_ROLES[i % len(EVAL_ROLES)])
        for i in range(args.sessions)
    ]

    print(f"Running {len(plan)} sessions against {MODEL_NAME}")
    print(f"Roughly {len(plan) * (MAX_TURNS + 2)} LLM calls. Ctrl-C to stop.\n")

    results: list[SessionResult] = []
    for index, (persona, role, level) in enumerate(plan, start=1):
        print(f"  [{index:>3}/{len(plan)}] {persona.name:<9} {role:<22} ", end="", flush=True)
        result = run_session(persona, role, level, args.pause)
        results.append(result)
        if result.ok:
            passed = sum(1 for c in result.checks if c["passed"])
            print(f"{passed}/{len(result.checks)} checks  {result.duration_seconds}s")
        else:
            print(f"ERROR  {result.error[:60]}")

    summary = summarise(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest.json").write_text(
        json.dumps({"summary": summary, "sessions": [asdict(r) for r in results]}, indent=2)
    )
    (out_dir / "latest.md").write_text(render_markdown(summary))

    print(f"\n{'=' * 62}")
    print(f"  sessions   {summary['sessions_completed']} completed, "
          f"{summary['sessions_errored']} errored")
    print(f"  checks     {summary['checks_passed']}/{summary['total_checks']} "
          f"({summary['overall_pass_rate']:.1%})")
    for name, stats in summary["by_check"].items():
        flag = "" if stats["deterministic"] else "  (heuristic)"
        print(f"    {name:<34} {stats['passed']:>3}/{stats['total']:<3} "
              f"{stats['rate']:>6.1%}{flag}")
    print(f"\n  written to {out_dir}/latest.json and latest.md")

    # Non-zero only if the harness could not run, not because the agent scored
    # below some arbitrary bar. This measures; it does not gate.
    return 1 if summary["sessions_completed"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
