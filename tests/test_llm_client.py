"""
Key rotation and failover.

The fakes raise what langchain-google-genai actually raises, not what it looks
like it should raise. An earlier version of these tests raised bare APIErrors,
which the library never surfaces for client errors - so they passed against a
classification bug that left 429 rotation completely dead.
"""

import pytest
from google.genai.errors import APIError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError

import backend.llm_client as llm_client
from backend.llm_client import LLMUnavailableError, api_error_code


def raw_error(code):
    error = APIError.__new__(APIError)
    Exception.__init__(error, f"HTTP {code}")
    error.code = code
    return error


def library_error(code):
    """A 4xx as the library reports it: wrapped, original only on __cause__."""
    if code >= 500:
        return raw_error(code)
    wrapped = ChatGoogleGenerativeAIError(f"Error calling model (HTTP {code})")
    wrapped.__cause__ = raw_error(code)
    return wrapped


class KeyedClient:
    def __init__(self, index, behaviour, log):
        self.index = index
        self.behaviour = behaviour
        self.log = log

    def invoke(self, messages):
        self.log.append(self.index)
        error = self.behaviour(self.index)
        if error:
            raise error
        return AIMessage(content=f"response from key {self.index + 1}")


@pytest.fixture
def keys(monkeypatch):
    """Install three fake keys; the test supplies the failure behaviour."""
    log = []

    def install(behaviour, sticky=0):
        monkeypatch.setattr(
            llm_client, "_clients", [KeyedClient(i, behaviour, log) for i in range(3)]
        )
        monkeypatch.setattr(llm_client, "_sticky_index", sticky)
        return log

    return install


def test_first_key_answers_without_fallback(keys):
    log = keys(lambda i: None)
    assert llm_client.generate_response("hi") == "response from key 1"
    assert log == [0]


def test_rate_limited_key_falls_through(keys):
    log = keys(lambda i: library_error(429) if i == 0 else None)
    assert llm_client.generate_response("hi") == "response from key 2"
    assert log == [0, 1]
    assert llm_client._sticky_index == 1


def test_exhausted_key_is_not_retried_first(keys):
    """The sticky index is the whole reason with_fallbacks alone is not enough."""
    log = keys(lambda i: library_error(429) if i == 0 else None, sticky=1)
    llm_client.generate_response("hi")
    assert log == [1], "should skip the known-dead key entirely"


def test_fallback_order_wraps_around(keys):
    log = keys(lambda i: library_error(429) if i in (2, 0) else None, sticky=2)
    assert llm_client.generate_response("hi") == "response from key 2"
    assert log == [2, 0, 1]


def test_all_keys_exhausted_raises(keys):
    log = keys(lambda i: library_error(429))
    with pytest.raises(LLMUnavailableError) as caught:
        llm_client.generate_response("hi")
    assert log == [0, 1, 2]
    assert caught.value.__cause__ is not None


def test_malformed_request_does_not_burn_keys(keys):
    """400 is our bug. Falling back would hide it behind 'all keys exhausted'."""
    log = keys(lambda i: library_error(400))
    with pytest.raises(LLMUnavailableError, match="400"):
        llm_client.generate_response("hi")
    assert log == [0]


def test_unknown_model_does_not_burn_keys(keys):
    log = keys(lambda i: library_error(404))
    with pytest.raises(LLMUnavailableError, match="404"):
        llm_client.generate_response("hi")
    assert log == [0]


def test_server_error_is_retryable(keys):
    log = keys(lambda i: library_error(503) if i == 0 else None)
    assert llm_client.generate_response("hi") == "response from key 2"
    assert log == [0, 1]


def test_status_is_read_off_the_cause_chain():
    """ChatGoogleGenerativeAIError is not an APIError subclass; isinstance misses it."""
    wrapped = library_error(429)
    assert not isinstance(wrapped, APIError)
    assert api_error_code(wrapped) == 429


def test_non_api_exception_propagates_untouched(keys):
    keys(lambda i: TypeError("bug in our own code"))
    with pytest.raises(TypeError):
        llm_client.generate_response("hi")


def test_empty_model_output_raises(monkeypatch):
    class Blank:
        def invoke(self, messages):
            return AIMessage(content="   ")

    monkeypatch.setattr(llm_client, "_clients", [Blank()])
    with pytest.raises(LLMUnavailableError, match="empty"):
        llm_client.generate_response("hi")


def test_missing_keys_raise_with_an_actionable_message(monkeypatch):
    monkeypatch.setattr(llm_client, "_clients", [])
    with pytest.raises(LLMUnavailableError, match="GEMINI_API_KEY"):
        llm_client.generate_response("hi")


def test_empty_message_list_is_rejected(fake_llm):
    with pytest.raises(ValueError):
        llm_client.generate([])


def test_system_instruction_becomes_a_system_message(fake_llm):
    llm_client.generate_response("the prompt", system_instruction="the rules")
    kinds = [type(m).__name__ for m in fake_llm.last_messages]
    assert kinds == ["SystemMessage", "HumanMessage"]


def test_no_system_message_when_omitted(fake_llm):
    llm_client.generate_response("just a prompt")
    assert [type(m).__name__ for m in fake_llm.last_messages] == ["HumanMessage"]


def test_duplicate_keys_are_deduplicated(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "same")
    monkeypatch.setenv("GEMINI_API_KEY_2", "same")
    monkeypatch.setenv("GEMINI_API_KEY_3", "different")
    assert llm_client._load_keys() == ["same", "different"]
