"""
Shared test fixtures.

Two things have to happen before any backend module is imported, which is why
the environment is set at the very top of this file rather than in a fixture:

- ``DATA_DIR`` is read by backend.config at import time, and backend.graph opens
  its checkpointer connection at import time. Setting it later would leave the
  tests writing into the developer's real checkpoint database.
- ``GEMINI_API_KEY`` must exist so backend.llm_client constructs a client list.
  The client is replaced with a fake before anything is called, so the value is
  never used for a real request.
"""

import os
import tempfile
import uuid

os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="interview-tests-")
os.environ["GEMINI_API_KEY"] = "test-key-not-used"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from backend.graph import interview_graph, thread_config  # noqa: E402
from backend.schemas import InterviewFeedback, InterviewTurn  # noqa: E402
import backend.llm_client as llm_client  # noqa: E402


class FakeLLM:
    """
    Stands in for a ChatGoogleGenerativeAI client.

    Records what it was sent and returns whatever the test queued, so assertions
    can be made about the messages that reached the model as well as the result.
    """

    def __init__(self):
        self.calls = 0
        self.last_messages = None
        self.turn = InterviewTurn(action="advance", reply="What database have you used?")
        self.feedback = InterviewFeedback(
            overall_performance="Solid.",
            strengths=["Clear communication"],
            areas_for_improvement=["More depth on tradeoffs"],
            key_recommendations=["Practise system design"],
            verdict="Hire",
            verdict_reason="Good fundamentals.",
        )
        self.text = "Hello! Tell me about yourself."
        self.detection = None

    # Unstructured path: /start_session's greeting.
    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages
        return AIMessage(content=self.text)

    def with_structured_output(self, schema):
        outer = self

        class _Structured:
            def invoke(self, messages):
                outer.calls += 1
                outer.last_messages = messages
                if schema is InterviewTurn:
                    return outer.turn
                if schema is InterviewFeedback:
                    return outer.feedback
                return outer.detection

        return _Structured()


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the client list so no test can reach the network."""
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "_clients", [fake])
    monkeypatch.setattr(llm_client, "_sticky_index", 0)
    return fake


@pytest.fixture(autouse=True)
def clean_database():
    """
    Empty the checkpoint store between tests.

    The checkpointer holds a single connection opened at import, so the database
    file cannot be swapped per test. Truncating gives the same isolation.
    """
    yield
    checkpoint_connection = interview_graph.checkpointer.conn
    tables = checkpoint_connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    for (name,) in tables:
        checkpoint_connection.execute(f"DELETE FROM {name}")
    checkpoint_connection.commit()


@pytest.fixture
def make_session():
    """
    Create a persisted session seeded with a transcript.

    Mirrors what /start_session does: graph state written straight into the
    checkpoint without running a node.
    """

    def _make(
        role="Backend Developer",
        experience_level="Junior",
        history=(("model", "Tell me about yourself."),),
        questions_asked=0,
        max_questions=15,
        resume_text="",
        in_intro_stage=False,
    ):
        session_id = str(uuid.uuid4())
        messages = [
            AIMessage(content=text) if speaker == "model" else _human(text)
            for speaker, text in history
        ]
        interview_graph.update_state(
            thread_config(session_id),
            {
                "role": role,
                "experience_level": experience_level,
                "resume_text": resume_text,
                "messages": messages,
                "questions_asked": questions_asked,
                "max_questions": max_questions,
                "persona": "normal",
                "end_requested": False,
                "is_over": False,
                "last_action": "",
                "feedback": None,
                "current_question_follow_ups": 0,
                "in_closing_stage": False,
                "paste_detected": False,
                "in_intro_stage": in_intro_stage,
            },
        )
        return session_id

    return _make


def _human(text):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=text)
