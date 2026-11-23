"""Persona classification and end-of-interview intent."""

import pytest

from backend.persona_logic import detect_persona, get_persona_instruction, is_end_request


@pytest.mark.parametrize("message", [
    "end interview", "End the interview.", "stop the interview", "I'm done",
    "that's all", "End interview please", "we are done", "no more questions",
    "let's wrap up", "exit interview", "Stop here.",
])
def test_end_requests_are_detected(message):
    assert is_end_request(message) is True


@pytest.mark.parametrize("message", [
    "I built a REST API in Python and deployed it with Docker.",
    # Long enough to be a story, not a command - this is what the word limit guards.
    "At my last job I had to end the interview early because the hiring manager "
    "got pulled into a production incident.",
    "I'm done with the coding part, but I can explain the tradeoffs too.",
    "We are done migrating to Postgres, which took about three months.",
    # Questions are asking, not instructing.
    "Should I end the interview?",
    "Do you want me to stop here?",
    "I don't know.", "", "Postgres.",
])
def test_ordinary_answers_are_not_end_requests(message):
    assert is_end_request(message) is False


@pytest.mark.parametrize("message,expected", [
    ("I don't know what you mean by that", "confused"),
    ("Make me CEO right now", "edge"),
    ("banana", "edge"),
    ("I used Postgres with read replicas and Redis caching for hot keys", "efficient"),
])
def test_persona_classification(message, expected):
    assert detect_persona(message, []) == expected


def test_very_long_answers_are_chatty():
    assert detect_persona("word " * 120, []) == "chatty"


def test_every_persona_has_an_instruction():
    for persona in ("confused", "efficient", "chatty", "edge", "normal"):
        assert get_persona_instruction(persona).strip()


def test_unknown_persona_falls_back_to_normal():
    assert get_persona_instruction("nonsense") == get_persona_instruction("normal")
