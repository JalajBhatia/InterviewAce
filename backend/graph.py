"""
The interview as an explicit state machine.

Why LangGraph and not an LCEL chain
-----------------------------------
LCEL (``prompt | llm | parser``) composes stateless pipelines: input in,
transformations, output out. An interview is not a pipeline. Each turn has to
decide - against state that persists across turns - whether the last answer was
good enough to move on, whether to probe, whether the candidate asked to stop,
and whether the question budget is spent. That is branching over persistent
state, which is what a graph expresses and a pipeline cannot.

The shape
---------
                    START
                      |
                 [ classify ]           persona + end-intent, no LLM call
                      |
              route_after_classify
                 /          \\
        [ close ]            [ conduct ]      one structured LLM call
             |                    |
            END                  END

Two things this buys that the previous imperative version could not have:

1. ``questions_asked`` counts questions, not turns. The conduct node returns a
   declared ``action``, and only ``advance`` increments the counter. Previously
   every user message incremented it, so a candidate who needed three
   clarifications on question one had already spent four of fifteen slots - and
   the prompt was simultaneously told "Question 5 of 15", which was false.

2. Termination is one decision in one place. There were previously two competing
   mechanisms: a hardcoded cutoff in the endpoint and an instruction in the
   system prompt telling the model to close after N questions. The hardcoded one
   always won, so the prompt's version was unreachable dead text.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .llm_client import generate_structured
from .persona_logic import detect_persona, get_persona_instruction, is_closing_decline, is_end_request
from .prompts import ANSWER_EVALUATION_RULES, SYSTEM_PROMPT_TEMPLATE
from .schemas import AnswerEvaluation, InterviewTurn, RubricScores
from .rubric import sanitize_answer_evaluation
import prompts

CLOSING_MESSAGE = (
    "Thank you for your time today. That covers what I wanted to ask. "
    "Let me put together your feedback now."
)

CLOSING_ON_REQUEST = (
    "Of course - let's stop here. Thank you for your time today. "
    "Let me put together your feedback now."
)

CLOSING_QUESTION = (
    "Thank you for sharing your approach. We have reached the end of our "
    "technical interview questions today. Do you have any questions for me "
    "about the role or the team?"
)


class InterviewState(TypedDict):
    """
    Everything one turn needs to make its decisions.

    ``messages`` uses the ``add_messages`` reducer: a node returns only the new
    messages and the reducer appends them, rather than each node having to read,
    copy and rewrite the whole list. That is also the contract LangGraph
    checkpointers expect, which is what phase 3 will use for persistence.
    """

    role: str
    interview_type: str
    difficulty: str
    experience_level: str
    prompt_version: str
    resume_text: str
    messages: Annotated[list[BaseMessage], add_messages]
    questions_asked: int
    max_questions: int
    persona: str
    end_requested: bool
    is_over: bool
    last_action: str
    evaluations: list[dict]
    last_evaluation: dict | None
    feedback: dict | None
    current_question_follow_ups: int
    in_intro_stage: bool
    in_closing_stage: bool
    paste_detected: bool


def classify(state: InterviewState) -> dict:
    """
    Inspect the newest candidate message. Deterministic, no LLM call.

    Kept rule-based for the same reason persona detection always was: this runs
    on every turn of a voice app, where an extra round trip is latency the user
    hears, and both questions are cheap to answer directly.
    """
    messages = state["messages"]
    latest = messages[-1].text if messages else ""
    in_closing = state.get("in_closing_stage", False)
    end_req = is_end_request(latest) or (in_closing and is_closing_decline(latest))

    return {
        "persona": detect_persona(latest, _as_history(messages)),
        "end_requested": end_req,
        "paste_detected": bool(state.get("paste_detected", False)),
    }


def route_after_classify(state: InterviewState) -> Literal["close", "conduct"]:
    """
    The only place the interview can decide to end.

    Ordering matters: an explicit request to stop is honoured before the question
    budget is consulted, so asking to finish never depends on how far through the
    interview you are.
    """
    if state.get("end_requested"):
        return "close"
    if state.get("in_closing_stage"):
        return "close"
    if not state.get("in_intro_stage") and state.get("questions_asked", 0) >= state.get("max_questions", 5):
        return "close"
    return "conduct"


def conduct(state: InterviewState) -> dict:
    """Produce the next interviewer turn, and record what kind of turn it was."""
    resume = state.get("resume_text") or ""
    resume_part = f"\nCandidate Resume:\n{resume}\n" if resume else ""
    persona = state.get("persona", "normal")
    interview_type = state.get("interview_type", "Technical")
    difficulty = state.get("difficulty", "Medium")
    prompt_ver = state.get("prompt_version", "v4")
    in_intro = state.get("in_intro_stage", False)
    follow_ups = state.get("current_question_follow_ups", 0)
    max_q = state.get("max_questions", 5)
    current_q_num = min(state.get("questions_asked", 0) + 1, max_q)

    # Retrieve prompt configuration
    prompt_cfg = prompts.get_prompt_version(prompt_ver)
    system_template = prompt_cfg.get("system_prompt", SYSTEM_PROMPT_TEMPLATE)
    eval_rules = prompt_cfg.get("evaluation_prompt", ANSWER_EVALUATION_RULES)

    try:
        formatted_system = system_template.format(
            role=state["role"],
            interview_type=interview_type,
            difficulty=difficulty,
            experience_level=state["experience_level"],
            current_question_num=current_q_num,
            max_questions=max_q,
        )
    except KeyError:
        # Fallback if specific placeholder not present in legacy templates
        formatted_system = SYSTEM_PROMPT_TEMPLATE.format(
            role=state["role"],
            experience_level=state["experience_level"],
            current_question_num=current_q_num,
            max_questions=max_q,
        )

    if in_intro:
        stage_guidance = (
            f"\n\nSTAGE: MANDATORY INTRODUCTION COMPLETED.\n"
            f"The candidate has just answered the opening warm-up introduction question ('tell me about yourself and your background').\n"
            f"1. Warmly acknowledge the candidate's introduction and background in 1 concise sentence.\n"
            f"2. Immediately ask Technical Question 1 of {max_q} for the {state['role']} role ({interview_type} track, {difficulty} difficulty).\n"
            f"3. Evaluate the candidate's introduction response for clarity, relevance, and presentation (0-5 marks per criterion, total 20).\n"
            f"4. Set action to 'advance' to transition into the technical question set.\n"
            f"Do NOT repeat the introduction question."
        )
    else:
        followup_guidance = ""
        if follow_ups >= 1:
            followup_guidance = (
                "\n\nCRITICAL RULE: A follow-up question has already been asked for this topic. "
                "A maximum of 1 follow-up is allowed per question. You MUST NOT ask another follow-up. "
                "Choose action 'advance' to move forward to the next main question (or move to closing if question limit is reached)."
            )
        if current_q_num >= max_q:
            followup_guidance += (
                f"\n\nFINAL QUESTION NOTE: This is question {max_q} of {max_q} (the final technical question). "
                "If this question is completed or has already had a follow-up, advance to conclude the technical interview "
                f"and ask the candidate: '{CLOSING_QUESTION}'"
            )
        stage_guidance = followup_guidance

    system_instruction = (
        formatted_system
        + resume_part
        + f"\n\nDETECTED PERSONA: {persona.upper()}\n{get_persona_instruction(persona)}"
        + f"\n{eval_rules}"
        + stage_guidance
    )

    turn = generate_structured(
        [SystemMessage(content=system_instruction), *state["messages"]],
        InterviewTurn,
    )

    # Sanitize and validate evaluation server-side
    target_q_num = 0 if in_intro else current_q_num
    is_pasted = bool(state.get("paste_detected", False))
    if is_pasted:
        sanitized_eval = AnswerEvaluation(
            rubric=RubricScores(
                correctness=0,
                relevance=0,
                clarity=0,
                completeness=0,
                total_score=0,
                percentage=0.0,
            ),
            strengths=["No credit: pasted content detected."],
            areas_for_improvement=[
                "Pasted answer detected. Please answer in your own words to receive a score."
            ],
            suggested_improved_answer=(
                turn.evaluation.suggested_improved_answer
                if turn.evaluation and turn.evaluation.suggested_improved_answer
                else "Provide an original explanation in your own words articulating the key concepts, tradeoffs, and examples."
            ),
            follow_up_question=turn.evaluation.follow_up_question if turn.evaluation else "",
            question_number=target_q_num,
            paste_detected=True,
        )
    else:
        sanitized_eval = sanitize_answer_evaluation(turn.evaluation) if turn.evaluation else None
        if sanitized_eval:
            sanitized_eval.question_number = target_q_num
            sanitized_eval.paste_detected = False
    eval_dict = sanitized_eval.model_dump() if sanitized_eval else None

    current_evals = list(state.get("evaluations") or [])
    if eval_dict:
        current_evals.append(eval_dict)

    if in_intro:
        return {
            "messages": [AIMessage(content=turn.reply)],
            "questions_asked": 0,
            "last_action": "advance",
            "evaluations": current_evals,
            "last_evaluation": eval_dict,
            "is_over": False,
            "current_question_follow_ups": 0,
            "in_intro_stage": False,
            "in_closing_stage": False,
            "paste_detected": False,
        }

    action = turn.action
    reply = turn.reply

    if action == "follow_up":
        if follow_ups < 1:
            new_follow_ups = follow_ups + 1
            asked = state.get("questions_asked", 0)
            in_closing = False
        else:
            # Enforce max 1 follow-up per main question: force advance
            action = "advance"
            new_follow_ups = 0
            asked = state.get("questions_asked", 0) + 1
            in_closing = (asked >= max_q)
            if in_closing:
                if "questions for me" not in reply.lower() and "any questions" not in reply.lower():
                    reply = CLOSING_QUESTION
    elif action == "advance":
        new_follow_ups = 0
        asked = state.get("questions_asked", 0) + 1
        in_closing = (asked >= max_q)
        if in_closing:
            if "questions for me" not in reply.lower() and "any questions" not in reply.lower():
                reply = CLOSING_QUESTION
    else:  # rephrase or redirect
        new_follow_ups = follow_ups
        asked = state.get("questions_asked", 0)
        in_closing = False

    return {
        "messages": [AIMessage(content=reply)],
        "questions_asked": asked,
        "last_action": action,
        "evaluations": current_evals,
        "last_evaluation": eval_dict,
        "is_over": False,
        "current_question_follow_ups": new_follow_ups,
        "in_intro_stage": False,
        "in_closing_stage": in_closing,
        "paste_detected": False,
    }


def close(state: InterviewState) -> dict:
    """End the interview, with wording that matches why it ended."""
    messages = state.get("messages", [])
    latest = messages[-1].text if messages else ""

    if state.get("end_requested"):
        if state.get("in_closing_stage") and is_closing_decline(latest):
            text = CLOSING_MESSAGE
        elif is_closing_decline(latest) and not any(k in latest.lower() for k in ("end", "stop", "quit", "exit")):
            text = CLOSING_MESSAGE
        else:
            text = CLOSING_ON_REQUEST
    elif state.get("in_closing_stage"):
        if "?" in latest and not is_closing_decline(latest):
            text = (
                "That's a great question! Our engineering teams emphasize collaboration, "
                "continuous learning, and practical mentorship. Thank you for your time today! "
                "Let me put together your feedback now."
            )
        else:
            text = CLOSING_MESSAGE
    else:
        text = CLOSING_MESSAGE

    return {
        "messages": [AIMessage(content=text)],
        "is_over": True,
        "last_action": "close",
        "last_evaluation": None,
        "in_intro_stage": False,
        "in_closing_stage": True,
        "paste_detected": False,
    }


def _as_history(messages: list[BaseMessage]) -> list[dict]:
    """Adapt messages back to the dict shape detect_persona already expects."""
    return [
        {
            "role": "model" if isinstance(message, AIMessage) else "user",
            "content": message.text,
        }
        for message in messages
    ]


def build_checkpointer() -> SqliteSaver:
    """
    Persist graph state to SQLite so a restart does not destroy a live interview.

    This is the only store. There is no session table and no history: nothing
    identifies a user and nothing survives beyond the interview being taken.

    check_same_thread=False because the saver is a single long-lived object shared
    across FastAPI's threadpool. SqliteSaver serialises its own access, so the
    connection is never actually used concurrently - the flag disables a check
    that would otherwise reject a safe pattern.
    """
    import sqlite3
    import tempfile
    from pathlib import Path

    from .config import CHECKPOINT_PATH

    try:
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(CHECKPOINT_PATH, check_same_thread=False)
    except OSError:
        # Fallback to system temporary directory if target is read-only (e.g. Vercel /var/task)
        fallback_dir = Path(tempfile.gettempdir()) / "interviewace"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(fallback_dir / "checkpoints.db", check_same_thread=False)

    try:
        connection.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass
    return SqliteSaver(connection)


def build_interview_graph(checkpointer=None):
    """
    Compile the graph.

    With a checkpointer attached, state is saved after every node and restored by
    thread_id on the next invoke. That is what lets /chat send only the new
    message and have the whole conversation, question count and role context come
    back - and what lets an interview survive a server restart.
    """
    builder = StateGraph(InterviewState)

    builder.add_node("classify", classify)
    builder.add_node("conduct", conduct)
    builder.add_node("close", close)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify, ["close", "conduct"])
    builder.add_edge("conduct", END)
    builder.add_edge("close", END)

    return builder.compile(checkpointer=checkpointer)


def thread_config(session_id: str) -> dict:
    """Checkpoint key for a session. The session id IS the graph thread id."""
    return {"configurable": {"thread_id": session_id}}


interview_graph = build_interview_graph(build_checkpointer())
