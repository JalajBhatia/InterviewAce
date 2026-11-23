"""
Session lifecycle.

An interview lives only in the checkpoint, for the duration of the interview.
There is no session table, no listing and nothing retained across visits - the
checkpoint exists so a restart or a dropped connection does not destroy an
interview in progress.
"""

import pytest
from fastapi import HTTPException

from backend.graph import interview_graph, thread_config
from backend.main import (
    ChatRequest, StartSessionRequest, chat, load_state, start_session,
    transcript_from_checkpoint,
)


def state_of(session_id):
    return interview_graph.get_state(thread_config(session_id)).values


def run_turns(session_id, *messages):
    for message in messages:
        chat(ChatRequest(session_id=session_id, user_message=message))


def test_start_session_persists_a_row_and_seeds_the_checkpoint(fake_llm):
    result = start_session(StartSessionRequest(role="Backend Developer",
                                               experience_level="Senior"))
    session_id = result["session_id"]

    state = state_of(session_id)
    assert state["role"] == "Backend Developer"
    assert state["questions_asked"] == 0

    history = transcript_from_checkpoint(session_id)
    assert len(history) == 1, "greeting should be the only turn"
    assert history[0]["role"] == "model"


def test_greeting_does_not_run_the_graph(fake_llm):
    """
    start_session() initializes the fixed MANDATORY_INTRO_QUESTION directly without
    calling the LLM or executing graph nodes.
    """
    from backend.main import MANDATORY_INTRO_QUESTION

    calls_before = fake_llm.calls
    result = start_session(StartSessionRequest(role="Data Analyst", experience_level="Junior"))
    # Fixed mandatory opening question is seeded directly without calling the LLM
    assert fake_llm.calls == calls_before
    assert result["initial_message"] == MANDATORY_INTRO_QUESTION
    assert result["in_intro_stage"] is True

    # Graph is not executed during initialization: state has exactly 1 initial message
    state = state_of(result["session_id"])
    assert len(state["messages"]) == 1
    assert state["messages"][0].content == MANDATORY_INTRO_QUESTION
    assert state["questions_asked"] == 0
    assert state["in_intro_stage"] is True
    assert state["is_over"] is False


def test_transcript_accumulates_across_turns(fake_llm):
    """
    Verify transcript accumulation and question counter across warm-up introduction
    and technical questions:
    - Introduction is asked exactly once.
    - Introduction answer evaluates as warm-up and does not increment questions_asked (remains 0).
    - Interviewer asks Technical Question 1 after introduction answer.
    - Subsequent technical answers increment questions_asked correctly.
    """
    from backend.main import MANDATORY_INTRO_QUESTION

    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you for the intro! Question 1: How do you design a REST API?",
    })

    session_id = start_session(
        StartSessionRequest(role="Backend Developer", experience_level="Junior")
    )["session_id"]

    # Initial state: mandatory intro asked exactly once
    history = transcript_from_checkpoint(session_id)
    assert len(history) == 1
    assert history[0]["content"] == MANDATORY_INTRO_QUESTION
    assert state_of(session_id)["in_intro_stage"] is True
    assert state_of(session_id)["questions_asked"] == 0

    # Turn 1: Candidate answers introduction
    chat(ChatRequest(session_id=session_id, user_message="I build APIs."))
    state_after_intro = state_of(session_id)
    # Introduction does not increment questions_asked; transitions out of intro stage
    assert state_after_intro["in_intro_stage"] is False
    assert state_after_intro["questions_asked"] == 0
    history_after_intro = transcript_from_checkpoint(session_id)
    assert len(history_after_intro) == 3  # intro prompt, candidate answer, technical Q1
    # Verify the first technical question is asked after introduction without repeating intro
    assert "tell me about yourself" not in history_after_intro[2]["content"].lower()

    # Turn 2: Candidate answers Technical Question 1
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: How do you scale Postgres?",
    })
    chat(ChatRequest(session_id=session_id, user_message="Postgres with read replicas."))
    state_after_q1 = state_of(session_id)
    assert state_after_q1["questions_asked"] == 1
    assert len(transcript_from_checkpoint(session_id)) == 5

    # Turn 3: Candidate answers Technical Question 2
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3: How do you handle cache invalidation?",
    })
    chat(ChatRequest(session_id=session_id, user_message="Redis for hot keys."))
    state_after_q2 = state_of(session_id)
    assert state_after_q2["questions_asked"] == 2
    # greeting + intro answer + technical Q1 + Q1 answer + technical Q2 + Q2 answer + technical Q3
    assert len(transcript_from_checkpoint(session_id)) == 7


def test_state_is_read_from_the_checkpoint_not_memory(fake_llm):
    """
    Verify state restoration from checkpoint:
    - Candidate answers introduction (questions_asked=0, in_intro_stage=False).
    - Candidate answers Technical Question 1 (questions_asked=1).
    - Checkpoint accurately reflects role, experience level, question count, and transcript.
    """
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Nice background! Question 1: How do you run an A/B test?",
    })

    session_id = start_session(
        StartSessionRequest(role="Product Manager", experience_level="Mid-Level")
    )["session_id"]

    # Turn 1: Candidate answers introduction
    chat(ChatRequest(session_id=session_id, user_message="I owned the checkout funnel."))
    state1 = state_of(session_id)
    assert state1["in_intro_stage"] is False
    assert state1["questions_asked"] == 0

    # Turn 2: Candidate answers Technical Question 1
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: How do you define North Star metrics?",
    })
    chat(ChatRequest(session_id=session_id, user_message="We ran an A/B test."))

    state2 = state_of(session_id)
    assert state2["role"] == "Product Manager"
    assert state2["experience_level"] == "Mid-Level"
    # 1 technical question answered (intro was warm-up)
    assert state2["questions_asked"] == 1
    # greeting + intro answer + technical Q1 + technical Q1 answer + technical Q2
    assert len(transcript_from_checkpoint(session_id)) == 5


def test_an_interrupted_interview_resumes_where_it_stopped(fake_llm):
    """
    Verify that an interrupted interview resumes correctly from checkpoint:
    - Intro answered (in_intro_stage transitions to False).
    - Technical Q1 answered (questions_asked=1).
    - State reloaded from SQLite checkpoint.
    - Turn 3 resumes without repeating introduction or resetting question counter.
    """
    from backend.main import MANDATORY_INTRO_QUESTION

    session_id = start_session(
        StartSessionRequest(role="DevOps Engineer", experience_level="Senior")
    )["session_id"]

    # Turn 1: Candidate answers introduction
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 1: Explain Kubernetes cluster federation.",
    })
    chat(ChatRequest(session_id=session_id, user_message="I run Kubernetes in three regions."))
    assert state_of(session_id)["in_intro_stage"] is False
    assert state_of(session_id)["questions_asked"] == 0

    # Turn 2: Candidate answers Technical Question 1
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: How do you manage infrastructure drift?",
    })
    chat(ChatRequest(session_id=session_id, user_message="Terraform for provisioning."))
    assert state_of(session_id)["questions_asked"] == 1

    # Simulate interruption: reload state directly from checkpoint
    saved_state = load_state(session_id)
    assert saved_state["role"] == "DevOps Engineer"
    assert saved_state["in_intro_stage"] is False
    assert saved_state["questions_asked"] == 1

    # Turn 3: Interview resumes without repeating introduction or resetting question counter
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3: How do you secure CI pipelines?",
    })
    resp = chat(ChatRequest(session_id=session_id, user_message="And GitHub Actions for CI."))
    resumed_state = state_of(session_id)
    assert resumed_state["questions_asked"] == 2, "counter continued, not reset"
    assert resumed_state["in_intro_stage"] is False
    assert len(transcript_from_checkpoint(session_id)) == 7

    # Verify introduction was not repeated in interviewer turns
    ai_messages = [m["content"] for m in transcript_from_checkpoint(session_id) if m["role"] == "model"]
    assert ai_messages[0] == MANDATORY_INTRO_QUESTION
    for subsequent_ai_msg in ai_messages[1:]:
        assert "tell me about yourself and your background" not in subsequent_ai_msg.lower()


def test_only_the_new_message_is_sent_but_the_model_sees_everything(fake_llm):
    session_id = start_session(
        StartSessionRequest(role="Backend Developer", experience_level="Junior")
    )["session_id"]
    run_turns(session_id, "First answer.", "Second answer.")

    # /chat passes one HumanMessage; the checkpointer restores the rest.
    assert len(fake_llm.last_messages) >= 5


def test_chat_on_unknown_session_is_404(fake_llm):
    with pytest.raises(HTTPException) as caught:
        chat(ChatRequest(session_id="does-not-exist", user_message="hello"))
    assert caught.value.status_code == 404


def test_there_is_no_session_listing_endpoint():
    """History was removed on purpose: nothing is retained across visits."""
    from backend import main

    assert not hasattr(main, "list_sessions")
    assert not hasattr(main, "get_session_detail")


def test_unknown_session_state_is_none(fake_llm):
    assert load_state("does-not-exist") is None
