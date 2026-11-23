"""
Unit tests for InterviewAce Rubric Scoring, Backend Validation,
Performance Report Analytics, and Prompt Engineering Versions.
"""

import pytest
from pydantic import ValidationError

from backend.rubric import (
    clamp_score,
    compute_performance_summary,
    sanitize_answer_evaluation,
    validate_and_compute_rubric,
)
from backend.schemas import AnswerEvaluation, PerformanceSummary, RubricScores
import prompts


def test_clamp_score_bounds():
    """Scores must always stay strictly within [0, 5]."""
    assert clamp_score(-10) == 0
    assert clamp_score(-1) == 0
    assert clamp_score(0) == 0
    assert clamp_score(3) == 3
    assert clamp_score(5) == 5
    assert clamp_score(8) == 5
    assert clamp_score(100) == 5
    assert clamp_score("4") == 4
    assert clamp_score("invalid") == 0
    assert clamp_score(None) == 0


def test_validate_and_compute_rubric_overrides_fake_totals():
    """Backend must recompute total score and percentage rather than trusting model."""
    fake_ai_output = {
        "correctness": 5,
        "relevance": 4,
        "clarity": 4,
        "completeness": 3,
        "total_score": 999,  # Hallucinated total
        "percentage": 100.0,  # Hallucinated percentage
    }
    rubric = validate_and_compute_rubric(fake_ai_output)

    assert rubric.correctness == 5
    assert rubric.relevance == 4
    assert rubric.clarity == 4
    assert rubric.completeness == 3
    # 5 + 4 + 4 + 3 = 16
    assert rubric.total_score == 16
    # (16 / 20) * 100 = 80.0%
    assert rubric.percentage == 80.0


def test_validate_and_compute_rubric_handles_out_of_bounds_criteria():
    """Out-of-bounds criteria are clamped before summing."""
    extreme_data = {
        "correctness": 10,  # Clamped to 5
        "relevance": -3,   # Clamped to 0
        "clarity": 4,      # 4
        "completeness": 2, # 2
    }
    rubric = validate_and_compute_rubric(extreme_data)
    assert rubric.correctness == 5
    assert rubric.relevance == 0
    assert rubric.clarity == 4
    assert rubric.completeness == 2
    assert rubric.total_score == 11
    assert rubric.percentage == 55.0


def test_sanitize_answer_evaluation_handles_none_and_malformed():
    """Ensure malformed or empty evaluation objects are safely sanitized."""
    assert sanitize_answer_evaluation(None) is None
    assert sanitize_answer_evaluation("not a dict") is None

    partial = {
        "rubric": {"correctness": 4, "relevance": 4, "clarity": 4, "completeness": 4},
        "strengths": ["Clear articulation"],
        "areas_for_improvement": [],  # Empty
        "suggested_improved_answer": "Model answer",
        "follow_up_question": "Follow-up question?",
    }
    sanitized = sanitize_answer_evaluation(partial)
    assert isinstance(sanitized, AnswerEvaluation)
    assert sanitized.rubric.total_score == 16
    assert len(sanitized.strengths) == 1
    assert len(sanitized.areas_for_improvement) >= 1  # Fallback added


def test_compute_performance_summary_with_empty_evaluations():
    """Handle edge case where candidate answered zero questions."""
    summary = compute_performance_summary([])
    assert isinstance(summary, PerformanceSummary)
    assert summary.questions_answered == 0
    assert summary.total_marks_obtained == 0
    assert summary.max_possible_marks == 0
    assert summary.overall_percentage == 0.0
    assert summary.average_score == 0.0
    assert summary.strongest_criterion == "N/A"
    assert summary.weakest_criterion == "N/A"


def test_compute_performance_summary_accurate_aggregation():
    """Verify criteria totals, averages, strongest/weakest, and overall percentage."""
    eval1 = AnswerEvaluation(
        rubric=RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
        strengths=["Strong DSA knowledge"],
        areas_for_improvement=["Minor edge case"],
        suggested_improved_answer="Exemplar 1",
        follow_up_question="Q1 probe?",
    )
    eval2 = AnswerEvaluation(
        rubric=RubricScores(correctness=3, relevance=4, clarity=3, completeness=2, total_score=12, percentage=60.0),
        strengths=["Good relevance"],
        areas_for_improvement=["Depth is lacking"],
        suggested_improved_answer="Exemplar 2",
        follow_up_question="Q2 probe?",
    )

    summary = compute_performance_summary([eval1, eval2])

    assert summary.questions_answered == 2
    # Marks: 18 + 12 = 30
    assert summary.total_marks_obtained == 30
    # Max marks: 2 * 20 = 40
    assert summary.max_possible_marks == 40
    # Overall percentage: (30 / 40) * 100 = 75.0%
    assert summary.overall_percentage == 75.0
    # Average score: 30 / 2 = 15.0
    assert summary.average_score == 15.0

    # Criterion totals:
    # Correctness: 5 + 3 = 8
    # Relevance: 5 + 4 = 9 (Highest)
    # Clarity: 4 + 3 = 7
    # Completeness: 4 + 2 = 6 (Lowest)
    assert summary.criterion_totals["correctness"] == 8
    assert summary.criterion_totals["relevance"] == 9
    assert summary.criterion_totals["clarity"] == 7
    assert summary.criterion_totals["completeness"] == 6

    assert "Relevance" in summary.strongest_criterion
    assert "Completeness" in summary.weakest_criterion


def test_prompt_version_registry():
    """Ensure all required prompt versions (v1, v2, v3, v4) exist and load cleanly."""
    versions = ["v1", "v2", "v3", "v4"]
    for v in versions:
        cfg = prompts.get_prompt_version(v)
        assert "system_prompt" in cfg
        assert "evaluation_prompt" in cfg
        assert len(cfg["system_prompt"]) > 50

    # Fallback to default
    default_cfg = prompts.get_prompt_version("unknown_v99")
    assert default_cfg == prompts.PROMPT_VERSIONS[prompts.DEFAULT_PROMPT_VERSION]


def test_start_session_with_custom_interviewace_fields(fake_llm):
    """Test start_session with Milestone 1 fields: track, difficulty, count, prompt version."""
    from backend.main import StartSessionRequest, start_session, load_state

    req = StartSessionRequest(
        role="Software Development Intern",
        interview_type="Technical",
        difficulty="Hard",
        experience_level="Junior",
        question_count=8,
        prompt_version="v4",
    )
    result = start_session(req)
    session_id = result["session_id"]
    state = load_state(session_id)

    assert state["role"] == "Software Development Intern"
    assert state["interview_type"] == "Technical"
    assert state["difficulty"] == "Hard"
    assert state["prompt_version"] == "v4"
    assert state["max_questions"] == 8
    assert state["questions_asked"] == 0
    assert state["evaluations"] == []


def test_chat_turn_returns_rubric_evaluation(fake_llm, make_session):
    """Ensure /chat response encapsulates AnswerEvaluation and running progress."""
    from backend.main import ChatRequest, chat

    session_id = make_session(role="Software Development Intern")

    # Set mock turn with evaluation
    mock_rubric = RubricScores(
        correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0
    )
    mock_eval = AnswerEvaluation(
        rubric=mock_rubric,
        strengths=["Clear logic"],
        areas_for_improvement=["Optimize memory"],
        suggested_improved_answer="Exemplar",
        follow_up_question="Next question?",
    )
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "What is the time complexity?",
        "evaluation": mock_eval,
    })

    resp = chat(ChatRequest(session_id=session_id, user_message="My answer about lists and tuples."))

    assert resp.agent_message == "What is the time complexity?"
    assert resp.evaluation is not None
    assert resp.evaluation.rubric.total_score == 18
    assert resp.evaluation.rubric.percentage == 90.0
    assert resp.evaluation.rubric.correctness == 5
    assert resp.action == "advance"


def test_feedback_endpoint_includes_performance_summary(fake_llm, make_session):
    """Ensure /feedback populates performance_summary with actual session metrics."""
    from backend.graph import interview_graph, thread_config
    from backend.main import FeedbackRequest, get_feedback

    session_id = make_session(
        role="Frontend Developer",
        history=(("model", "Explain CSS specificity."), ("user", "Classes have weight 10, IDs have weight 100.")),
    )

    # Seed an evaluation into session state
    mock_eval = {
        "rubric": {"correctness": 4, "relevance": 5, "clarity": 4, "completeness": 3, "total_score": 16, "percentage": 80.0},
        "strengths": ["Accurate specificity weights"],
        "areas_for_improvement": ["Mention inline styles"],
        "suggested_improved_answer": "Inline styles have weight 1000.",
        "follow_up_question": "What about !important?",
    }
    interview_graph.update_state(thread_config(session_id), {"evaluations": [mock_eval]})

    res = get_feedback(FeedbackRequest(session_id=session_id))
    fb = res.feedback

    assert fb.performance_summary is not None
    assert fb.performance_summary.questions_answered == 1
    assert fb.performance_summary.total_marks_obtained == 16
    assert fb.performance_summary.max_possible_marks == 20
    assert fb.performance_summary.overall_percentage == 80.0


def test_regression_score_aggregation_with_followups_prevents_142_over_100():
    """
    Regression Test for Issue 1 (142/100 bug):
    In a 5-question interview with multiple follow-up turns (e.g., 8 turns total),
    scores must be aggregated per question slot. Each question contributes at most 20 marks,
    and the total score obtained must NEVER exceed max_questions * 20 (100).
    """
    # 8 turn evaluations representing 5 questions with follow-ups on Q1, Q2, and Q4
    # The raw, unaggregated sum of these 8 turns is EXACTLY 142 (18+19+17+18+18+16+18+18 = 142).
    # In the buggy code, the session score displayed 142 / 100.
    turn_evals = [
        # Question 1: Initial answer (18/20) and Follow-up answer (19/20)
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
            question_number=1,
            strengths=["Strong start on fundamentals"],
        ),
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=4, total_score=19, percentage=95.0),
            question_number=1,
            strengths=["Follow-up answered completely"],
        ),
        # Question 2: Initial answer (17/20) and Follow-up answer (18/20)
        AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=4, completeness=4, total_score=17, percentage=85.0),
            question_number=2,
            strengths=["Solid approach"],
        ),
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=4, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=2,
            strengths=["Follow-up clarified runtime complexity"],
        ),
        # Question 3: 18/20 (no follow-up)
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=4, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=3,
            strengths=["Solid OOP explanation"],
        ),
        # Question 4: Initial answer (16/20) and Follow-up answer (18/20)
        AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=4, total_score=16, percentage=80.0),
            question_number=4,
            strengths=["Basic SQL understanding"],
        ),
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
            question_number=4,
            strengths=["Resolved indexing trade-offs"],
        ),
        # Question 5: 18/20 (no follow-up)
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=4, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=5,
            strengths=["Strong concluding response"],
        ),
    ]

    # Verify that the unaggregated turn scores sum to exactly 142 (reproducing the original bug)
    raw_unaggregated_sum = sum(e.rubric.total_score for e in turn_evals)
    assert raw_unaggregated_sum == 142, f"Raw sum expected 142, got {raw_unaggregated_sum}"

    # With question-slot resolution, follow-up answers update their question slot:
    # Q1: 19, Q2: 18, Q3: 18, Q4: 18, Q5: 18 => Sum = 91 / 100
    summary = compute_performance_summary(turn_evals, max_questions=5)

    assert summary.questions_answered == 5
    # Max possible marks for 5 questions is strictly 100 (never 142 or 160)
    assert summary.max_possible_marks == 100
    # Resolved scores per question: Q1=19, Q2=18, Q3=18, Q4=18, Q5=18 => Sum = 91
    assert summary.total_marks_obtained == 91
    assert summary.total_marks_obtained <= 100
    # Overall percentage: (91 / 100) * 100 = 91.0%
    assert summary.overall_percentage == 91.0
    # Average score per question: 91 / 5 = 18.2 (<= 20)
    assert summary.average_score == 18.2
    # Each criterion total is <= 25 (5 questions * 5 marks)
    for crit, tot in summary.criterion_totals.items():
        assert tot <= 25, f"{crit} total {tot} exceeds max 25"
    # Each criterion average is <= 5.0
    for crit, avg in summary.criterion_averages.items():
        assert avg <= 5.0, f"{crit} average {avg} exceeds max 5.0"


def test_regression_neutral_greeting_and_no_assumed_candidate_name():
    """
    Regression Test for Issue 2 ('Shorya' name hallucination):
    Ensure prompts strictly enforce neutral greetings and do NOT assume or invent
    candidate names when none is provided.
    """
    from backend.prompts import INITIAL_QUESTION_PROMPT, SYSTEM_PROMPT_TEMPLATE
    import prompts

    # 1. No prompt text in backend or prompts/ should hardcode 'Shorya'
    assert "shorya" not in INITIAL_QUESTION_PROMPT.lower()
    assert "shorya" not in SYSTEM_PROMPT_TEMPLATE.lower()

    v4_cfg = prompts.get_prompt_version("v4")
    assert "shorya" not in v4_cfg["system_prompt"].lower()

    # 2. INITIAL_QUESTION_PROMPT must mandate neutral greeting and forbid name invention
    assert "neutral" in INITIAL_QUESTION_PROMPT.lower()
    assert "never assume" in INITIAL_QUESTION_PROMPT.lower() or "do not assume" in INITIAL_QUESTION_PROMPT.lower()

    # 3. Verify start_session prompt formatting with and without candidate_name
    from backend.main import StartSessionRequest

    req_no_name = StartSessionRequest(role="Software Development Intern", candidate_name=None)
    assert req_no_name.candidate_name is None

    req_with_name = StartSessionRequest(role="Software Development Intern", candidate_name="Aarav Sharma")
    assert req_with_name.candidate_name == "Aarav Sharma"


def test_voice_features_frontend_integration():
    """
    Verify voice input (STT) and voice output (TTS) contract in frontend files.
    Ensures HTML elements, JS Web Speech API integration, error recovery,
    watchdog timer, and replay buttons are present and correctly wired.
    """
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    html_file = repo_root / "frontend" / "index.html"
    js_file = repo_root / "frontend" / "script.js"
    css_file = repo_root / "frontend" / "style.css"

    assert html_file.exists(), "index.html not found"
    assert js_file.exists(), "script.js not found"
    assert css_file.exists(), "style.css not found"

    html_content = html_file.read_text(encoding="utf-8")
    js_content = js_file.read_text(encoding="utf-8")
    css_content = css_file.read_text(encoding="utf-8")

    # 1. Voice Input (STT) HTML Elements
    assert 'id="mode-voice-btn"' in html_content
    assert 'id="voice-mode-area"' in html_content
    assert 'id="mic-icon"' in html_content
    assert 'id="live-transcript"' in html_content
    assert 'id="stop-listening-btn"' in html_content

    # 2. Voice Input (STT) JavaScript Logic
    assert "SpeechRecognition" in js_content
    assert "webkitSpeechRecognition" in js_content
    assert "toggleVoiceRecognition" in js_content
    assert "startListening" in js_content
    assert "stopListening" in js_content
    assert "isListening" in js_content
    assert "continuous = true" in js_content
    assert "interimResults = true" in js_content

    # 3. Voice Output (TTS) JavaScript Logic
    assert "speechSynthesis" in js_content
    assert "SpeechSynthesisUtterance" in js_content
    assert "function speak(" in js_content
    assert "SPEECH_WATCHDOG_GRACE_MS" in js_content
    assert "replay-voice-btn" in js_content
    assert "window._activeUtterance" in js_content

    # 4. Voice CSS Styles
    assert ".mic-circle" in css_content
    assert ".recording" in css_content
    assert ".replay-voice-btn" in css_content


def test_rubric_scoring_all_edge_cases():
    """
    Test all rubric scoring edge cases required by the audit:
    - No evaluations
    - One question
    - Multiple questions
    - Multiple follow-up evaluations for the same question
    - Missing or invalid question numbers
    - Maximum score boundary
    """
    # 1. No evaluations
    summary_empty = compute_performance_summary([])
    assert summary_empty.questions_answered == 0
    assert summary_empty.total_marks_obtained == 0
    assert summary_empty.max_possible_marks == 0
    assert summary_empty.overall_percentage == 0.0

    # 2. One question
    eval_q1 = AnswerEvaluation(
        rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=4, total_score=16, percentage=80.0),
        question_number=1,
    )
    summary_one = compute_performance_summary([eval_q1])
    assert summary_one.questions_answered == 1
    assert summary_one.total_marks_obtained == 16
    assert summary_one.max_possible_marks == 20
    assert summary_one.overall_percentage == 80.0

    # 3. Multiple questions
    eval_q2 = AnswerEvaluation(
        rubric=RubricScores(correctness=5, relevance=4, clarity=4, completeness=4, total_score=17, percentage=85.0),
        question_number=2,
    )
    eval_q3 = AnswerEvaluation(
        rubric=RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
        question_number=3,
    )
    summary_multi = compute_performance_summary([eval_q1, eval_q2, eval_q3], max_questions=3)
    assert summary_multi.questions_answered == 3
    assert summary_multi.total_marks_obtained == 51
    assert summary_multi.max_possible_marks == 60
    assert summary_multi.overall_percentage == 85.0

    # 4. Multiple follow-up evaluations for the same question
    eval_q1_attempt1 = AnswerEvaluation(
        rubric=RubricScores(correctness=2, relevance=3, clarity=3, completeness=2, total_score=10, percentage=50.0),
        question_number=1,
    )
    eval_q1_attempt2 = AnswerEvaluation(
        rubric=RubricScores(correctness=3, relevance=4, clarity=4, completeness=3, total_score=14, percentage=70.0),
        question_number=1,
    )
    eval_q1_attempt3 = AnswerEvaluation(
        rubric=RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
        question_number=1,
    )
    summary_followup = compute_performance_summary([eval_q1_attempt1, eval_q1_attempt2, eval_q1_attempt3], max_questions=1)
    # The question's score is updated, NOT accumulated (10 + 14 + 18 = 42 would exceed max 20)
    assert summary_followup.questions_answered == 1
    assert summary_followup.total_marks_obtained == 18
    assert summary_followup.max_possible_marks == 20
    assert summary_followup.total_marks_obtained <= 20

    # 5. Missing or invalid question numbers
    unindexed_1 = {
        "rubric": {"correctness": 3, "relevance": 3, "clarity": 3, "completeness": 3, "total_score": 12, "percentage": 60.0},
        "question_number": None,
    }
    unindexed_2 = {
        "rubric": {"correctness": 4, "relevance": 4, "clarity": 4, "completeness": 4, "total_score": 16, "percentage": 80.0},
        "question_number": -99,  # Negative
    }
    unindexed_3 = {
        "rubric": {"correctness": 5, "relevance": 4, "clarity": 4, "completeness": 4, "total_score": 17, "percentage": 85.0},
        "question_number": "invalid",  # String non-integer
    }
    summary_unindexed = compute_performance_summary([unindexed_1, unindexed_2, unindexed_3], max_questions=3)
    assert summary_unindexed.questions_answered == 3
    assert summary_unindexed.total_marks_obtained == 45
    assert summary_unindexed.max_possible_marks == 60
    assert summary_unindexed.total_marks_obtained <= 60

    # 6. Maximum score boundary
    perfect_evals = [
        AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=i,
        )
        for i in range(1, 6)
    ]
    summary_perfect = compute_performance_summary(perfect_evals, max_questions=5)
    assert summary_perfect.questions_answered == 5
    assert summary_perfect.total_marks_obtained == 100
    assert summary_perfect.max_possible_marks == 100
    assert summary_perfect.overall_percentage == 100.0
    assert summary_perfect.total_marks_obtained <= summary_perfect.max_possible_marks


def test_regression_single_opening_question_and_empty_answer_prevention(fake_llm):
    """
    Regression Test for Issue 5:
    1. Ensures INITIAL_QUESTION_PROMPT mandates asking exactly ONE question.
    2. Ensures start_session sets exactly 1 initial message in checkpoint.
    3. Ensures /chat rejects empty/whitespace answers with HTTP 400 to prevent duplicate loops.
    """
    from fastapi import HTTPException
    from backend.prompts import INITIAL_QUESTION_PROMPT
    from backend.main import ChatRequest, StartSessionRequest, chat, load_state, start_session

    # 1. Prompt instruction checks
    prompt_lower = INITIAL_QUESTION_PROMPT.lower()
    assert "exactly one" in prompt_lower
    assert "do not ask multiple" in prompt_lower

    # 2. Start session creates exactly 1 message
    result = start_session(StartSessionRequest(role="Software Development Intern"))
    sid = result["session_id"]
    state = load_state(sid)
    assert len(state["messages"]) == 1
    assert state["questions_asked"] == 0
    assert result["initial_message"]

    # 3. Empty message rejection in /chat
    with pytest.raises(HTTPException) as exc_info:
        chat(ChatRequest(session_id=sid, user_message="   "))
    assert exc_info.value.status_code == 400
    assert "empty answer" in exc_info.value.detail.lower()


# ==============================================================================
# Regression Tests for Question Count and Closing Stage Evaluation Behavior
# ==============================================================================

def test_regression_selected_question_count_controls_main_questions(fake_llm, make_session):
    """
    1. Selected question count (e.g. 3) controls number of main questions.
    2. Follow-up does not spend main question count.
    3. Exactly 3 main questions are conducted before transitioning to closing stage.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)

    # Q1 Answer: LLM asks follow-up
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "follow_up",
        "reply": "Can you elaborate on your linked list deletion logic?",
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="Arrays are contiguous, linked lists use pointers."))
    state1 = interview_graph.get_state(thread_config(session_id)).values
    assert state1["questions_asked"] == 0, "Follow-up must not increment main question counter"
    assert state1["current_question_follow_ups"] == 1
    assert state1["in_closing_stage"] is False
    assert resp1.action == "follow_up"

    # Q1 Follow-up Answer: LLM advances to Q2
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: Explain SQL indexing.",
    })
    resp2 = chat(ChatRequest(session_id=session_id, user_message="Deletion is O(1) if pointer is known."))
    state2 = interview_graph.get_state(thread_config(session_id)).values
    assert state2["questions_asked"] == 1, "Completed Q1, moving to Q2"
    assert state2["current_question_follow_ups"] == 0
    assert state2["in_closing_stage"] is False
    assert resp2.action == "advance"

    # Q2 Answer: LLM advances to Q3
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3: Explain REST vs GraphQL.",
    })
    resp3 = chat(ChatRequest(session_id=session_id, user_message="B-Tree indexes speed up lookups at cost of slower writes."))
    state3 = interview_graph.get_state(thread_config(session_id)).values
    assert state3["questions_asked"] == 2, "Completed Q2, moving to Q3"
    assert state3["current_question_follow_ups"] == 0
    assert state3["in_closing_stage"] is False

    # Q3 Answer: Last main question completed -> Must transition directly to closing stage
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you. That concludes our technical questions. Do you have any questions for me?",
    })
    resp4 = chat(ChatRequest(session_id=session_id, user_message="REST uses fixed endpoints, GraphQL allows client-specified queries."))
    state4 = interview_graph.get_state(thread_config(session_id)).values
    assert state4["questions_asked"] == 3, "All 3 main questions completed"
    assert state4["in_closing_stage"] is True
    assert "questions for me" in resp4.agent_message.lower()


def test_regression_max_one_follow_up_per_main_question(fake_llm, make_session):
    """
    Allow a maximum of 1 relevant follow-up per main question.
    If LLM attempts a 2nd follow-up on the same main question, backend forces advance.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "follow_up",
        "reply": "First follow-up question.",
    })

    # Turn 1: 1st follow-up is allowed
    resp1 = chat(ChatRequest(session_id=session_id, user_message="Initial answer."))
    assert resp1.action == "follow_up"
    state1 = interview_graph.get_state(thread_config(session_id)).values
    assert state1["questions_asked"] == 0
    assert state1["current_question_follow_ups"] == 1

    # Turn 2: LLM tries to follow up again -> Backend forces advance
    resp2 = chat(ChatRequest(session_id=session_id, user_message="Follow-up answer."))
    assert resp2.action == "advance", "Second follow-up must be forced to advance"
    state2 = interview_graph.get_state(thread_config(session_id)).values
    assert state2["questions_asked"] == 1, "Must advance to question 2"
    assert state2["current_question_follow_ups"] == 0


def test_regression_closing_question_not_evaluated_with_technical_rubric(fake_llm, make_session):
    """
    The closing question ('Do you have any questions for me?') must not be evaluated
    using the technical rubric.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=3)
    # Seed state in closing stage
    interview_graph.update_state(
        thread_config(session_id),
        {"in_closing_stage": True, "questions_asked": 3, "evaluations": []},
    )

    resp = chat(ChatRequest(session_id=session_id, user_message="no thank you"))

    assert resp.is_interview_over is True
    assert resp.evaluation is None, "Closing question response must NOT receive technical evaluation"
    assert resp.action == "close"
    state = interview_graph.get_state(thread_config(session_id)).values
    assert len(state.get("evaluations", [])) == 0, "No evaluation should be appended for closing response"
    assert state.get("last_evaluation") is None


@pytest.mark.parametrize("closing_phrase", [
    "no thank you",
    "no thanks",
    "no, thank you",
    "no, thanks!",
    "No thank you so much.",
    "nope, all good",
    "I don't have any questions, thank you!",
    "no questions from my side",
    "All clear, thanks",
    "none, thank you",
    "no",
])
def test_regression_closing_response_phrases_accepted_and_end_interview(fake_llm, make_session, closing_phrase):
    """
    If candidate answers 'no thank you' or similar, accept it as a normal closing
    response and end the interview cleanly.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=3)
    interview_graph.update_state(
        thread_config(session_id),
        {"in_closing_stage": True, "questions_asked": 3},
    )

    resp = chat(ChatRequest(session_id=session_id, user_message=closing_phrase))

    assert resp.is_interview_over is True
    assert resp.evaluation is None
    assert "thank you for your time" in resp.agent_message.lower()


def test_regression_no_reuse_of_previous_technical_evaluation_on_closing(fake_llm, make_session):
    """
    Do not reuse the previous technical evaluation or exemplar answer for the closing response.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=2)

    # Question 3 technical evaluation
    mock_eval = AnswerEvaluation(
        rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
        strengths=["Perfect system design"],
        areas_for_improvement=["Minor edge case"],
        suggested_improved_answer="Exemplar answer for Question 3",
        question_number=3,
    )
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "We have reached the end of technical questions. Do you have any questions for me?",
        "evaluation": mock_eval,
    })

    # Answer Q3
    resp_q3 = chat(ChatRequest(session_id=session_id, user_message="My thorough answer to Question 3."))
    assert resp_q3.evaluation is not None
    assert resp_q3.evaluation.question_number == 3
    assert resp_q3.evaluation.suggested_improved_answer == "Exemplar answer for Question 3"
    assert resp_q3.is_interview_over is False

    # Candidate responds to closing question with "no thank you"
    resp_closing = chat(ChatRequest(session_id=session_id, user_message="no thank you"))

    # Crucial assertion: Closing turn must NOT have an evaluation or reuse Q3 exemplar
    assert resp_closing.is_interview_over is True
    assert resp_closing.evaluation is None, "Must NOT reuse previous evaluation or exemplar"
    state = interview_graph.get_state(thread_config(session_id)).values
    assert state.get("last_evaluation") is None


def test_regression_final_report_contains_only_valid_technical_evaluations(fake_llm, make_session):
    """
    Ensure the final report contains only valid technical answer evaluations and the
    correct question count (e.g. 3 questions answered, max marks 60, no closing evaluations).
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, FeedbackRequest, chat, get_feedback

    session_id = make_session(max_questions=3, questions_asked=0)

    # Conduct Q1 (18/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=1,
            strengths=["Q1 strength"],
        ),
    })
    chat(ChatRequest(session_id=session_id, user_message="Answer 1"))

    # Conduct Q2 (17/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=5, completeness=4, total_score=17, percentage=85.0),
            question_number=2,
            strengths=["Q2 strength"],
        ),
    })
    chat(ChatRequest(session_id=session_id, user_message="Answer 2"))

    # Conduct Q3 (19/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "We have concluded technical questions. Do you have any questions for me?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=4, total_score=19, percentage=95.0),
            question_number=3,
            strengths=["Q3 strength"],
        ),
    })
    chat(ChatRequest(session_id=session_id, user_message="Answer 3"))

    # Answer closing question
    chat(ChatRequest(session_id=session_id, user_message="no thank you"))

    # Fetch final performance summary report
    fb_resp = get_feedback(FeedbackRequest(session_id=session_id))
    summary = fb_resp.feedback.performance_summary

    assert summary is not None
    assert summary.questions_answered == 3, f"Expected 3 questions answered, got {summary.questions_answered}"
    assert summary.max_possible_marks == 60, f"Expected 60 max marks, got {summary.max_possible_marks}"
    # Total marks: 18 + 17 + 19 = 54
    assert summary.total_marks_obtained == 54
    # (54 / 60) * 100 = 90.0%
    assert summary.overall_percentage == 90.0
    assert summary.average_score == 18.0
    assert len(summary.overall_strengths) > 0


def test_regression_frontend_question_counter_and_closing_guards():
    """Verify frontend/script.js implements closing rubric guards and question counter bounds."""
    from pathlib import Path

    js_file = Path(__file__).resolve().parent.parent / "frontend" / "script.js"
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")

    # Guard ensuring closing turns never render rubric cards
    assert "!data.is_interview_over && data.action !== \"close\"" in content
    # Guard ensuring currentQuestionIndex does not advance beyond totalQuestions
    assert "currentQuestionIndex < totalQuestions" in content
    # Topbar display when completed
    assert "Completed (${totalQuestions} of ${totalQuestions})" in content


# ==============================================================================
# COPY / PASTE INTEGRITY REGRESSION TESTS
# ==============================================================================

def test_pasted_answer_receives_zero_score_and_warning(fake_llm, make_session):
    """
    Requirements 3, 4, 5, 6, 7:
    When paste_detected=True is sent with candidate answer:
    - Backend strictly overrides score to 0/20 (correctness=0, relevance=0, clarity=0, completeness=0).
    - Evaluation contains exact warning: 'Pasted answer detected. Please answer in your own words to receive a score.'
    - paste_detected flag is True on evaluation.
    - Session total score reflects 0 marks.
    """
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)

    # Even if LLM would evaluate this as a perfect 20/20 answer
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Moving on to question 2.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=1,
            strengths=["Superb explanation copied from web"],
            areas_for_improvement=["None"],
        ),
    })

    resp = chat(ChatRequest(
        session_id=session_id,
        user_message="def binary_search(arr, target): return bisect_left(arr, target)",
        paste_detected=True,
    ))

    assert resp.evaluation is not None
    assert resp.evaluation.paste_detected is True
    # Rubric must be 0/20
    assert resp.evaluation.rubric.total_score == 0
    assert resp.evaluation.rubric.percentage == 0.0
    assert resp.evaluation.rubric.correctness == 0
    assert resp.evaluation.rubric.relevance == 0
    assert resp.evaluation.rubric.clarity == 0
    assert resp.evaluation.rubric.completeness == 0
    # Must display clear warning message
    expected_msg = "Pasted answer detected. Please answer in your own words to receive a score."
    assert any(expected_msg in item for item in resp.evaluation.areas_for_improvement)
    assert resp.session_total_score == 0


def test_normal_typed_answer_scores_normally(fake_llm, make_session):
    """
    Requirements 7, 12:
    When paste_detected=False (normal typing), candidate answer is evaluated normally
    without 0-score penalty or paste warnings.
    """
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)

    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Moving on to question 2.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=5, completeness=4, total_score=17, percentage=85.0),
            question_number=1,
            strengths=["Clear articulation and solid algorithmic intuition"],
            areas_for_improvement=["Consider edge case where array is empty"],
        ),
    })

    resp = chat(ChatRequest(
        session_id=session_id,
        user_message="I would initialize two pointers, left and right, and halve the search space at each iteration.",
        paste_detected=False,
    ))

    assert resp.evaluation is not None
    assert resp.evaluation.paste_detected is False
    assert resp.evaluation.rubric.total_score == 17
    assert resp.evaluation.rubric.percentage == 85.0
    assert not any("Pasted answer detected" in item for item in resp.evaluation.areas_for_improvement)
    assert resp.session_total_score == 17


def test_microphone_voice_answer_scores_normally(fake_llm, make_session):
    """
    Requirements 12:
    Microphone/voice recognition answers do not trigger paste events and default
    paste_detected=False. They receive full normal evaluation scores.
    """
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)

    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Great explanation.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=4, completeness=5, total_score=18, percentage=90.0),
            question_number=1,
            strengths=["Concise spoken answer"],
            areas_for_improvement=["Elaborate slightly on space complexity"],
        ),
    })

    # Voice input payload omits paste_detected (defaults False)
    resp = chat(ChatRequest(
        session_id=session_id,
        user_message="I would use a hash table to achieve O of 1 average lookup time.",
    ))

    assert resp.evaluation is not None
    assert resp.evaluation.paste_detected is False
    assert resp.evaluation.rubric.total_score == 18
    assert resp.session_total_score == 18


def test_paste_flag_resets_per_answer_turn(fake_llm, make_session):
    """
    Requirement 8:
    Reset the paste flag for every new answer so that one pasted answer does not
    affect the entire interview.
    Q1: Pasted -> 0/20 with warning.
    Q2: Genuine typed answer -> 19/20 normal score.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=3, questions_asked=0)

    # Q1: Pasted answer
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Here is question 2.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=1,
            strengths=["LLM output"],
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="pasted code", paste_detected=True))
    assert resp1.evaluation.rubric.total_score == 0
    assert resp1.evaluation.paste_detected is True

    # Graph state must have reset paste_detected to False
    state1 = interview_graph.get_state(thread_config(session_id)).values
    assert state1.get("paste_detected") is False

    # Q2: Genuine typed answer (paste_detected=False)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Here is question 3.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=4, total_score=19, percentage=95.0),
            question_number=2,
            strengths=["Thorough explanation of ACID guarantees"],
            areas_for_improvement=["Mention isolation levels"],
        ),
    })
    resp2 = chat(ChatRequest(session_id=session_id, user_message="Atomicity ensures all-or-nothing execution.", paste_detected=False))
    assert resp2.evaluation.rubric.total_score == 19
    assert resp2.evaluation.paste_detected is False
    assert not any("Pasted answer detected" in item for item in resp2.evaluation.areas_for_improvement)
    # Total score: 0 + 19 = 19
    assert resp2.session_total_score == 19


def test_pasted_follow_up_answer_does_not_increase_score(fake_llm, make_session):
    """
    Requirement 9:
    If a follow-up answer is pasted, it must not increase the score for that question.
    Apply the existing question-slot scoring logic correctly.
    Q1 Main: Candidate gets 14/20.
    Q1 Follow-up: Candidate pastes answer -> receives 0/20.
    Question 1 score does not increase (stays <= 14).
    """
    from backend.main import ChatRequest, FeedbackRequest, chat, get_feedback

    session_id = make_session(max_questions=2, questions_asked=0)

    # Q1 Main Answer: Scores 14/20, interviewer probes with follow-up
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "follow_up",
        "reply": "Can you explain how you handle collisions in that hash table?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=3, relevance=4, clarity=4, completeness=3, total_score=14, percentage=70.0),
            question_number=1,
            strengths=["Understands hash map lookup"],
            areas_for_improvement=["Elaborate on collision resolution"],
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="I'd use a HashMap for key-value storage.", paste_detected=False))
    assert resp1.action == "follow_up"
    assert resp1.evaluation.rubric.total_score == 14
    assert resp1.session_total_score == 14

    # Q1 Follow-up Answer: Candidate PASTES answer from internet
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Moving to question 2.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=1,
        ),
    })
    resp2 = chat(ChatRequest(
        session_id=session_id,
        user_message="Separate chaining uses linked lists while open addressing probes next slots.",
        paste_detected=True,
    ))

    # Follow-up evaluation is strictly 0/20 with warning
    assert resp2.evaluation.rubric.total_score == 0
    assert resp2.evaluation.paste_detected is True
    assert any("Pasted answer detected" in item for item in resp2.evaluation.areas_for_improvement)
    # Question slot score must not increase
    assert resp2.session_total_score <= 14


def test_closing_question_not_evaluated_with_rubric_even_if_paste_sent(fake_llm, make_session):
    """
    Requirements 10, 11:
    - Do not apply 0/20 scoring rule or technical rubric to closing question.
    - If candidate answers closing question (even if paste flag was true), evaluation is None.
    - Do not reuse previous answer evaluation or exemplar answer.
    """
    from backend.main import ChatRequest, FeedbackRequest, chat, get_feedback

    session_id = make_session(max_questions=1, questions_asked=0)

    # Q1 Answer -> Advance to closing question
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "We have reached the end of our technical interview questions today. Do you have any questions for me about the role or the team?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=4, clarity=4, completeness=4, total_score=17, percentage=85.0),
            question_number=1,
            strengths=["Solid technical answer"],
            suggested_improved_answer="Here is an exemplar model answer.",
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="My technical answer.", paste_detected=False))
    assert resp1.evaluation.rubric.total_score == 17

    # Closing Response: Candidate responds (even if paste_detected=True)
    resp2 = chat(ChatRequest(session_id=session_id, user_message="What is the engineering culture like?", paste_detected=True))

    assert resp2.is_interview_over is True
    # Evaluation must be None (not 0/20, not reused)
    assert resp2.evaluation is None

    # Performance summary contains strictly 1 question answered
    fb = get_feedback(FeedbackRequest(session_id=session_id))
    summary = fb.feedback.performance_summary
    assert summary.questions_answered == 1
    assert summary.total_marks_obtained == 17
    assert summary.max_possible_marks == 20


def test_frontend_copy_paste_integrity_controls():
    """
    Requirements 1, 2, 5, 6, 8:
    Verify frontend HTML, CSS, and JS files have full copy/paste detection implementation.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    # 1. HTML Banner check
    html_file = root / "frontend" / "index.html"
    assert html_file.exists()
    html_text = html_file.read_text(encoding="utf-8")
    assert "id=\"paste-warning-banner\"" in html_text
    assert "Pasted answer detected." in html_text
    assert "Please answer in your own words to receive a score." in html_text

    # 2. CSS styles check
    css_file = root / "frontend" / "style.css"
    assert css_file.exists()
    css_text = css_file.read_text(encoding="utf-8")
    assert ".paste-warning-banner" in css_text
    assert ".paste-warning-banner.hidden" in css_text
    assert ".paste-eval-alert" in css_text

    # 3. JS Controller check
    js_file = root / "frontend" / "script.js"
    assert js_file.exists()
    js_text = js_file.read_text(encoding="utf-8")
    assert "initPasteDetection" in js_text
    assert "markPasteDetected" in js_text
    assert "resetPasteDetection" in js_text
    assert "paste_detected: Boolean(" in js_text or "paste_detected: Boolean(pasteDetected)" in js_text
    # Check keyboard and paste event handlers
    assert "addEventListener(\"paste\"" in js_text
    assert "addEventListener(\"keydown\"" in js_text
    assert "isCtrlOrCmdV" in js_text or "ctrlKey" in js_text
    assert "Shift" in js_text or "shiftKey" in js_text
    assert "resetPasteDetection()" in js_text


# ==============================================================================
# COMPLETION FLOW, TIMER, AND FINAL REPORT REGRESSION TESTS
# ==============================================================================

def test_end_to_end_fresh_3_question_interview_completion(fake_llm, make_session):
    """
    Requirements 1, 2, 3, 4, 7:
    Manually/programmatically test a fresh 3-question interview:
    1. Q1 main answer: evaluated, score recorded.
    2. Q2 main answer: evaluated, score recorded.
    3. Q3 main answer: evaluated, advances to closing stage (asking closing question).
    4. Closing response ('no thank you'):
       - Backend explicitly returns is_interview_over=True, is_completed=True, action='close'.
       - evaluation is None (closing turn is not evaluated with rubric).
       - question counter remains strictly at 3.
    5. Final report generated via /feedback:
       - Contains exactly 3 questions answered and max 60 marks.
       - Performance summary percentage and scores are calculated accurately.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, FeedbackRequest, chat, get_feedback

    session_id = make_session(max_questions=3, questions_asked=0)

    # Turn 1: Candidate answers Q1 (18/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: Explain database indexing.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=1,
            strengths=["Strong algorithmic reasoning"],
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="My approach uses binary search in O(log N)."))
    assert resp1.questions_asked == 1
    assert resp1.is_interview_over is False
    assert resp1.is_completed is False
    assert resp1.evaluation.rubric.total_score == 18
    assert resp1.session_total_score == 18

    # Turn 2: Candidate answers Q2 (17/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3: Walk me through REST vs gRPC.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=5, completeness=4, total_score=17, percentage=85.0),
            question_number=2,
            strengths=["Good understanding of B-trees"],
        ),
    })
    resp2 = chat(ChatRequest(session_id=session_id, user_message="B-Tree indexes optimize range queries."))
    assert resp2.questions_asked == 2
    assert resp2.is_interview_over is False
    assert resp2.is_completed is False
    assert resp2.evaluation.rubric.total_score == 17
    assert resp2.session_total_score == 35

    # Turn 3: Candidate answers Q3 (19/20) -> Moves to closing stage
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "We have reached the end of our technical interview questions today. Do you have any questions for me about the role or the team?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=4, total_score=19, percentage=95.0),
            question_number=3,
            strengths=["Detailed comparison of serialization formats"],
        ),
    })
    resp3 = chat(ChatRequest(session_id=session_id, user_message="gRPC uses HTTP/2 and Protobuf for high throughput."))
    assert resp3.questions_asked == 3
    assert resp3.is_interview_over is False
    assert resp3.is_completed is False
    assert resp3.evaluation.rubric.total_score == 19
    assert resp3.session_total_score == 54

    state3 = interview_graph.get_state(thread_config(session_id)).values
    assert state3["in_closing_stage"] is True
    assert state3["questions_asked"] == 3

    # Turn 4: Candidate answers closing question ('no thank you')
    resp_close = chat(ChatRequest(session_id=session_id, user_message="No thank you, that covers everything!"))
    assert resp_close.is_interview_over is True
    assert resp_close.is_completed is True
    assert resp_close.action == "close"
    assert resp_close.evaluation is None, "Closing question must NOT have a technical evaluation"
    assert resp_close.questions_asked == 3, "Question count must remain strictly at selected count (3)"
    assert "Thank you for your time today" in resp_close.agent_message or "Let me put together your feedback" in resp_close.agent_message

    state_final = interview_graph.get_state(thread_config(session_id)).values
    assert state_final["is_over"] is True

    # Turn 5: Request Final Report via /feedback
    fb_resp = get_feedback(FeedbackRequest(session_id=session_id))
    summary = fb_resp.feedback.performance_summary

    assert summary is not None
    assert summary.questions_answered == 3, "Report must reflect exactly 3 technical questions answered"
    assert summary.max_possible_marks == 60, "Max marks must be 3 * 20 = 60"
    assert summary.total_marks_obtained == 54, "Total marks obtained must be 18 + 17 + 19 = 54"
    assert summary.overall_percentage == 90.0
    assert summary.average_score == 18.0
    assert len(summary.overall_strengths) > 0


def test_frontend_completion_timer_controls_and_retry_flow():
    """
    Requirements 2, 3, 5, 6, 8:
    Verify frontend/script.js implements:
    1. Authoritative completion check (is_interview_over || is_completed || action === 'close').
    2. Immediate stopTimer() call when completion is received.
    3. Input controls and submit buttons permanently disabled upon completion.
    4. Guard against duplicate report generation (isGeneratingReport).
    5. Retry button and retryGenerateReport() handler on error.
    6. Reset of completion and report state upon startInterview().
    """
    from pathlib import Path

    js_file = Path(__file__).resolve().parent.parent / "frontend" / "script.js"
    assert js_file.exists()
    js_text = js_file.read_text(encoding="utf-8")

    # Authoritative completion condition check
    assert "data.is_interview_over || data.is_completed || data.action === \"close\"" in js_text

    # Immediate timer and listening shutdown
    assert "stopTimer();" in js_text
    assert "stopListening();" in js_text

    # UI controls disabled
    assert "isInterviewCompleted = true;" in js_text
    assert "submitBtn.disabled = true;" in js_text
    assert "textInput.disabled = true;" in js_text

    # Progress text set to Completed
    assert "Completed (${totalQuestions} of ${totalQuestions})" in js_text

    # Guard against duplicate feedback requests
    assert "if (isGeneratingReport) return;" in js_text
    assert "isGeneratingReport = true;" in js_text

    # Retryable report error display
    assert "retryGenerateReport" in js_text
    assert "retry-report-btn" in js_text

    # StartInterview resets completion and report generating flags
    assert "isInterviewCompleted = false;" in js_text
    assert "isGeneratingReport = false;" in js_text


def test_clicking_finish_interview_does_not_create_duplicate_reports(fake_llm, make_session):
    """
    Requirements 4, 8:
    Clicking 'Finish Interview & Get Report' multiple times does not create
    duplicate reports or duplicate evaluations. Backend checkpoint caching guarantees
    subsequent calls return the identical report without calling the model again.
    """
    from backend.main import FeedbackRequest, get_feedback

    session_id = make_session(
        max_questions=2,
        questions_asked=1,
        history=(("model", "Tell me about yourself."), ("user", "I am a backend developer.")),
    )

    first_call = get_feedback(FeedbackRequest(session_id=session_id))
    calls_after_first = fake_llm.calls

    # Second call (simulating duplicate click or rapid user tap)
    second_call = get_feedback(FeedbackRequest(session_id=session_id))

    assert fake_llm.calls == calls_after_first, "Duplicate feedback request must NOT call LLM again"
    assert first_call.feedback.performance_summary.questions_answered == second_call.feedback.performance_summary.questions_answered
    assert first_call.feedback.performance_summary.total_marks_obtained == second_call.feedback.performance_summary.total_marks_obtained


# ==============================================================================
# MANDATORY INTRODUCTION & PASTE INTEGRITY REGRESSION TESTS (PARTS 1, 2, 3)
# ==============================================================================

def test_mandatory_introduction_begins_every_session(fake_llm):
    """
    Requirements Part 2 & Part 3 (8, 9, 10):
    Every new interview session begins with the exact fixed mandatory introduction question:
    'Hello, and welcome! To get started, could you please tell me about yourself and your background?'
    across any role, difficulty, and track.
    """
    from backend.main import MANDATORY_INTRO_QUESTION, StartSessionRequest, load_state, start_session

    roles = ["Software Development Intern", "Frontend Developer", "Machine Learning Intern", "HR Interview"]
    tracks = ["Technical", "HR"]
    difficulties = ["Easy", "Medium", "Hard"]

    for r in roles:
        for t in tracks:
            for d in difficulties:
                resp = start_session(StartSessionRequest(
                    role=r,
                    interview_type=t,
                    difficulty=d,
                    question_count=3,
                ))
                assert resp["initial_message"] == MANDATORY_INTRO_QUESTION
                assert resp["in_intro_stage"] is True

                state = load_state(resp["session_id"])
                assert state["in_intro_stage"] is True
                assert state["questions_asked"] == 0
                assert len(state["messages"]) == 1
                assert state["messages"][0].content == MANDATORY_INTRO_QUESTION


def test_introduction_asked_only_once_and_transitions_to_technical_q1(fake_llm):
    """
    Requirements Part 2 & Part 3 (9, 10, 11):
    1. Introduction is asked exactly once.
    2. After candidate answers, state transitions out of intro stage (in_intro_stage=False).
    3. Interviewer immediately asks Technical Question 1.
    4. questions_asked remains 0 (warm-up does not consume technical question count).
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import ChatRequest, StartSessionRequest, chat, start_session

    session_info = start_session(StartSessionRequest(
        role="Software Development Intern",
        interview_type="Technical",
        question_count=3,
    ))
    session_id = session_info["session_id"]

    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you for sharing your background! Let's begin: Can you explain how quicksort works and its worst-case complexity?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=4, completeness=4, total_score=17, percentage=85.0),
            question_number=0,
            strengths=["Clear communication of background"],
        ),
    })

    # Candidate answers introduction
    resp = chat(ChatRequest(
        session_id=session_id,
        user_message="I am a third-year CS student with internship experience in Python and cloud services.",
        paste_detected=False,
    ))

    state = interview_graph.get_state(thread_config(session_id)).values
    assert state["in_intro_stage"] is False, "Interview must exit intro stage after intro is answered"
    assert state["questions_asked"] == 0, "Introduction must not consume technical question count"
    assert "quicksort" in resp.agent_message.lower(), "Interviewer must ask Technical Question 1"
    assert "tell me about yourself" not in resp.agent_message.lower(), "Interviewer must not repeat introduction"
    assert resp.questions_asked == 0


def test_pasted_introduction_receives_zero_and_warning(fake_llm):
    """
    Requirements Part 2 & Part 3 (2, 3):
    A pasted introduction answer receives strictly 0/20 and paste warning,
    even if the mocked LLM returns 20/20.
    """
    from backend.main import ChatRequest, StartSessionRequest, chat, start_session

    session_info = start_session(StartSessionRequest(role="Backend Developer", question_count=3))
    session_id = session_info["session_id"]

    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Moving to technical question 1.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=0,
            strengths=["LLM output"],
        ),
    })

    resp = chat(ChatRequest(
        session_id=session_id,
        user_message="I have extensive experience with distributed systems and PostgreSQL copied from the web.",
        paste_detected=True,
    ))

    assert resp.evaluation is not None
    assert resp.evaluation.paste_detected is True
    assert resp.evaluation.rubric.total_score == 0
    assert resp.evaluation.rubric.percentage == 0.0
    assert any("Pasted answer detected" in msg for msg in resp.evaluation.areas_for_improvement)


def test_full_3_question_interview_with_mandatory_intro_and_closing(fake_llm):
    """
    Requirements Part 2 & Part 3 (8, 9, 10, 11, 12):
    Complete 3-question interview session:
    1. Start: Mandatory Introduction is asked.
    2. Turn 1 (Intro response): Evaluated as warm-up, transitions to Technical Q1.
    3. Turn 2 (Q1 response): Evaluated (18/20), questions_asked=1, asks Technical Q2.
    4. Turn 3 (Q2 response): Evaluated (17/20), questions_asked=2, asks Technical Q3.
    5. Turn 4 (Q3 response): Evaluated (19/20), questions_asked=3, asks Closing Question.
    6. Turn 5 (Closing response): Ends interview without technical evaluation.
    7. Final Report: Exactly 3 technical questions answered, max 60 marks, total 54 marks.
    """
    from backend.graph import interview_graph, thread_config
    from backend.main import (
        ChatRequest,
        FeedbackRequest,
        MANDATORY_INTRO_QUESTION,
        StartSessionRequest,
        chat,
        get_feedback,
        start_session,
    )

    # 1. Start Session
    start_resp = start_session(StartSessionRequest(
        role="Software Development Intern",
        interview_type="Technical",
        question_count=3,
    ))
    session_id = start_resp["session_id"]
    assert start_resp["initial_message"] == MANDATORY_INTRO_QUESTION

    # 2. Turn 1: Candidate answers introduction
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Great background! Let's start Question 1: How does hash collision resolution work?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=4, total_score=16, percentage=80.0),
            question_number=0,
            strengths=["Concise background overview"],
        ),
    })
    resp_intro = chat(ChatRequest(session_id=session_id, user_message="I'm a computer science senior passionate about backend APIs."))
    assert resp_intro.questions_asked == 0
    assert resp_intro.session_total_score == 0  # Technical running score starts at 0

    # 3. Turn 2: Candidate answers Technical Question 1 (18/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 2: Explain deadlock conditions and how to prevent them.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=5, clarity=5, completeness=4, total_score=18, percentage=90.0),
            question_number=1,
            strengths=["Clear chaining and probing explanations"],
        ),
    })
    resp_q1 = chat(ChatRequest(session_id=session_id, user_message="Chaining stores collided keys in linked lists."))
    assert resp_q1.questions_asked == 1
    assert resp_q1.session_total_score == 18

    # 4. Turn 3: Candidate answers Technical Question 2 (17/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Question 3: Describe how indexing speeds up queries in relational databases.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=5, completeness=4, total_score=17, percentage=85.0),
            question_number=2,
            strengths=["Accurate listing of Coffman conditions"],
        ),
    })
    resp_q2 = chat(ChatRequest(session_id=session_id, user_message="Coffman conditions: mutual exclusion, hold and wait, no preemption, circular wait."))
    assert resp_q2.questions_asked == 2
    assert resp_q2.session_total_score == 35

    # 5. Turn 4: Candidate answers Technical Question 3 (19/20) -> Concludes technical round
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you for sharing your approach. We have reached the end of our technical interview questions today. Do you have any questions for me about the role or the team?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=4, total_score=19, percentage=95.0),
            question_number=3,
            strengths=["Detailed B-tree and disk I/O analysis"],
        ),
    })
    resp_q3 = chat(ChatRequest(session_id=session_id, user_message="B-trees allow O(log N) lookups by minimizing disk page reads."))
    assert resp_q3.questions_asked == 3
    assert resp_q3.session_total_score == 54

    state_post_q3 = interview_graph.get_state(thread_config(session_id)).values
    assert state_post_q3["in_closing_stage"] is True

    # 6. Turn 5: Candidate responds to closing question
    resp_close = chat(ChatRequest(session_id=session_id, user_message="No thank you, that covers everything!"))
    assert resp_close.is_interview_over is True
    assert resp_close.is_completed is True
    assert resp_close.action == "close"
    assert resp_close.evaluation is None

    # 7. Final Report
    fb_resp = get_feedback(FeedbackRequest(session_id=session_id))
    summary = fb_resp.feedback.performance_summary
    assert summary.questions_answered == 3, f"Expected 3 questions answered, got {summary.questions_answered}"
    assert summary.max_possible_marks == 60, f"Expected 60 max marks, got {summary.max_possible_marks}"
    assert summary.total_marks_obtained == 54, f"Expected 54 marks, got {summary.total_marks_obtained}"
    assert summary.overall_percentage == 90.0
    assert summary.average_score == 18.0


def test_frontend_script_paste_lifecycle_and_input_preservation():
    """
    Requirements Part 1 B:
    Verify frontend/script.js:
    - Captures wasPasted = Boolean(isPasteDetected)
    - Does NOT reset paste flag or clear input before sending
    - Resets paste flag and clears input only after fetch succeeds
    - Resets paste flag when input is cleared to empty
    """
    from pathlib import Path

    js_file = Path(__file__).resolve().parent.parent / "frontend" / "script.js"
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")

    # In submitTextAnswer: captures wasPasted and delegates to sendUserResponse
    assert "const wasPasted = Boolean(isPasteDetected);" in content
    assert "sendUserResponse(answer, wasPasted);" in content

    # In sendUserResponse: sends paste_detected in body
    assert "paste_detected: Boolean(pasteDetected)" in content

    # In sendUserResponse: resetPasteDetection() only after await response.json()
    fetch_idx = content.find("const response = await fetch")
    json_idx = content.find("const data = await response.json()")
    reset_idx = content.find("resetPasteDetection();", json_idx)
    assert fetch_idx != -1 and json_idx != -1 and reset_idx != -1
    assert reset_idx > json_idx, "resetPasteDetection must only be called after response is parsed"

    # Input event resets when length === 0
    assert "if (textInput.value.trim().length === 0)" in content


def test_frontend_script_cumulative_score_and_session_maximum():
    """
    Requirements 3, 7, 8:
    Verify frontend/script.js:
    - Uses backend authoritative session_total_score and max_possible_score.
    - Displays session score against the session maximum.
    - Preserves previous slot scores when a follow-up answer is pasted.
    """
    from pathlib import Path

    js_file = Path(__file__).resolve().parent.parent / "frontend" / "script.js"
    assert js_file.exists()
    content = js_file.read_text(encoding="utf-8")

    assert "sessionTotalScore = (data.session_total_score !== undefined && data.session_total_score !== null)" in content
    assert "data.max_possible_score" in content
    assert "runningEl.textContent = `${sessionTotalScore} / ${maxScore}`;" in content
    assert "if (data.evaluation.paste_detected)" in content


def test_cumulative_scoring_at_question_2_with_pasted_answer(fake_llm, make_session):
    """
    Requirements 1-4, 7-8:
    - 5 questions selected. Each question is worth 20 marks (session max = 100).
    - Session score is cumulative across distinct technical questions.
    - At Question 2, if Q1 scored 15 and Q2 scored 0 (pasted), cumulative score is 15 / 100.
    - Pasted answers must remain 0 and must not accidentally reset cumulative score.
    """
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=5, questions_asked=0, in_intro_stage=True)

    # Turn 1: Warm-up introduction
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you! Let's start with Question 1: What is a deadlock?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=4, total_score=16, percentage=80.0),
            question_number=0,
            strengths=["Good communication"],
            areas_for_improvement=[],
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="I have 4 years of experience as a software engineer."))
    assert resp1.session_total_score == 0, "Warm-up introduction must not count toward technical score"
    assert resp1.max_possible_score == 100, "5 questions session maximum is 100"

    # Turn 2: Question 1 Answer (Scores 15/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Great explanation. Question 2: Explain database indexing.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=3, total_score=15, percentage=75.0),
            question_number=1,
            strengths=["Understands mutual exclusion and hold and wait"],
            areas_for_improvement=["Elaborate on prevention techniques"],
        ),
    })
    resp2 = chat(ChatRequest(session_id=session_id, user_message="A deadlock occurs when processes hold resources and wait for others in a cycle."))
    assert resp2.evaluation.rubric.total_score == 15
    assert resp2.session_total_score == 15, "After Q1, cumulative score must be 15"
    assert resp2.max_possible_score == 100, "Session maximum must consistently be 100"

    # Turn 3: Question 2 Answer (PASTED -> 0/20)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Thank you. Question 3: How does consistent hashing work?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=2,
            strengths=["Detailed B-tree breakdown"],
        ),
    })
    resp3 = chat(ChatRequest(
        session_id=session_id,
        user_message="Indexes use B-trees to provide logarithmic lookup time for table rows.",
        paste_detected=True,
    ))
    assert resp3.evaluation.rubric.total_score == 0, "Pasted answer must score 0"
    assert resp3.evaluation.paste_detected is True
    assert resp3.evaluation.question_number == 2
    # Crucial assertion: Cumulative score remains 15 (Q1=15 + Q2=0 = 15), NOT 0
    assert resp3.session_total_score == 15, f"Cumulative score at Q2 must be 15, got {resp3.session_total_score}"
    assert resp3.max_possible_score == 100, f"Session maximum must be 100, got {resp3.max_possible_score}"


def test_pasted_follow_up_does_not_reset_main_question_score(fake_llm, make_session):
    """
    Requirement 5 & 7:
    - Q1 main answer scores 15/20.
    - Q1 follow-up is pasted (0/20).
    - Question 1 score remains 15/20 and does not get wiped out or reset to 0.
    """
    from backend.main import ChatRequest, chat

    session_id = make_session(max_questions=5, questions_asked=0)

    # Q1 Main: Scores 15/20, interviewer probes with follow-up
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "follow_up",
        "reply": "How do you detect deadlocks dynamically?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=4, relevance=4, clarity=4, completeness=3, total_score=15, percentage=75.0),
            question_number=1,
            strengths=["Good definition of deadlock"],
            areas_for_improvement=["Provide detection mechanisms"],
        ),
    })
    resp1 = chat(ChatRequest(session_id=session_id, user_message="A deadlock occurs when processes hold resources and wait for others."))
    assert resp1.action == "follow_up"
    assert resp1.session_total_score == 15
    assert resp1.max_possible_score == 100

    # Q1 Follow-up: Pasted answer
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Moving to Question 2: What is an index?",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=1,
        ),
    })
    resp2 = chat(ChatRequest(
        session_id=session_id,
        user_message="Deadlocks can be detected using resource allocation graphs and cycle detection algorithms.",
        paste_detected=True,
    ))
    assert resp2.evaluation.rubric.total_score == 0
    assert resp2.evaluation.paste_detected is True
    # Follow-up being 0 must NOT wipe out Q1's score of 15
    assert resp2.session_total_score == 15, f"Expected session score to remain 15, got {resp2.session_total_score}"
    assert resp2.max_possible_score == 100


def test_final_scoring_after_5_questions(fake_llm, make_session):
    """
    Requirements 1, 4, 6:
    - 5 questions selected.
    - Q1: 16/20
    - Q2: 14/20
    - Q3: 18/20
    - Q4: 0/20 (pasted)
    - Q5: 20/20
    - Closing question: 0 marks (not evaluated)
    - Final report: exactly out of 100 (total = 68 / 100, 68.0%).
    """
    from backend.main import ChatRequest, FeedbackRequest, chat, get_feedback

    session_id = make_session(max_questions=5, questions_asked=0, in_intro_stage=True)

    # Intro (warm-up)
    fake_llm.turn = fake_llm.turn.model_copy(update={
        "action": "advance",
        "reply": "Let's begin with Question 1.",
        "evaluation": AnswerEvaluation(
            rubric=RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
            question_number=0,
        ),
    })
    r_intro = chat(ChatRequest(session_id=session_id, user_message="Hello, I am a backend developer."))
    assert r_intro.session_total_score == 0
    assert r_intro.max_possible_score == 100

    scores = [16, 14, 18, 0, 20]
    expected_cum = [16, 30, 48, 48, 68]

    score_rubrics = {
        16: RubricScores(correctness=4, relevance=4, clarity=4, completeness=4, total_score=16, percentage=80.0),
        14: RubricScores(correctness=4, relevance=4, clarity=3, completeness=3, total_score=14, percentage=70.0),
        18: RubricScores(correctness=5, relevance=5, clarity=4, completeness=4, total_score=18, percentage=90.0),
        0: RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0), # When pasted, backend overrides to 0
        20: RubricScores(correctness=5, relevance=5, clarity=5, completeness=5, total_score=20, percentage=100.0),
    }

    for q_idx in range(1, 6):
        sc = scores[q_idx - 1]
        is_paste = (sc == 0)
        rubric_obj = score_rubrics[sc]
        next_action = "advance"

        fake_llm.turn = fake_llm.turn.model_copy(update={
            "action": next_action,
            "reply": f"Question {q_idx + 1}" if q_idx < 5 else "Do you have any questions for me about the role or the team?",
            "evaluation": AnswerEvaluation(
                rubric=rubric_obj,
                question_number=q_idx,
            ),
        })
        resp = chat(ChatRequest(
            session_id=session_id,
            user_message=f"Answer for Question {q_idx}",
            paste_detected=is_paste,
        ))
        assert resp.session_total_score == expected_cum[q_idx - 1], f"At Q{q_idx}, expected {expected_cum[q_idx - 1]}, got {resp.session_total_score}"
        assert resp.max_possible_score == 100

    # Closing answer
    r_close = chat(ChatRequest(session_id=session_id, user_message="No, thank you for your time today."))
    assert r_close.is_completed is True
    assert r_close.evaluation is None, "Closing question must not be evaluated"

    # Final report
    fb_resp = get_feedback(FeedbackRequest(session_id=session_id))
    summary = fb_resp.feedback.performance_summary
    assert summary.questions_answered == 5, f"Expected 5 questions answered, got {summary.questions_answered}"
    assert summary.max_possible_marks == 100, f"Expected 100 max marks, got {summary.max_possible_marks}"
    assert summary.total_marks_obtained == 68, f"Expected 68 marks obtained, got {summary.total_marks_obtained}"
    assert summary.overall_percentage == 68.0
    assert summary.average_score == 13.6






