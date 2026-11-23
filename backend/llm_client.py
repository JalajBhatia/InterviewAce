"""
LLM transport layer.

Wraps Google Gemini behind LangChain, with failover across multiple API keys.

Two properties this module is built to guarantee:

1. **No shared mutable credential state.** Each API key gets its own
   ``ChatGoogleGenerativeAI`` instance. The previous implementation called the
   module-global ``genai.configure(api_key=...)`` before every request, which
   meant two concurrent requests could clobber each other's credentials and a
   request could go out on a different key than the one it accounted for.

2. **Failures raise, they never return.** The previous implementation returned
   apologetic strings ("I'm having trouble connecting...") on error. Callers
   could not distinguish those from real model output, so they were appended to
   the interview transcript as genuine interviewer turns and fed back into every
   later prompt. A transport failure must not become conversation state.
"""

from __future__ import annotations

import logging
import os
import threading

from dotenv import load_dotenv
from google.genai.errors import APIError
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini model name. Can be overridden via GEMINI_MODEL in .env or environment.
# Note: gemini-2.5-flash-lite and older models are retired (404 NOT_FOUND).
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Checked in order; the first is the default, the rest are failover keys.
KEY_ENV_VARS = ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3")

# HTTP status codes where retrying the *same* request on a *different* key has a
# real chance of succeeding:
#   401/403 - this key is invalid, revoked, or lacks permission
#   429     - this key is rate limited or out of daily quota
#   5xx     - transient server-side failure (handled separately, by range)
# Deliberately excluded: 400 (malformed request) and 404 (bad model name). Those
# are bugs in our own code. Falling back on them would burn every key in turn and
# report "all keys exhausted" while hiding the actual defect.
RETRYABLE_STATUS = frozenset({401, 403, 429})


class LLMUnavailableError(RuntimeError):
    """The request could not be served. Raised instead of returning error text."""


class _KeyFailure(Exception):
    """Internal signal: this key failed, try the next one. Never escapes this module."""


def _load_keys() -> list[str]:
    """Read keys from the environment, preserving order, dropping blanks and duplicates."""
    keys: list[str] = []
    seen: set[str] = set()
    for var in KEY_ENV_VARS:
        key = (os.getenv(var) or "").strip()
        # Duplicate keys would make the fallback chain longer without making it
        # more resilient: the second copy fails for the same reason as the first.
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


API_KEYS = _load_keys()

# One client per key. Credentials live on the instance, so there is no global
# state for concurrent requests to race over.
_clients: list[ChatGoogleGenerativeAI] = [
    ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=key) for key in API_KEYS
]

if not _clients:
    # Warn but stay importable: tests and CI import this module without keys.
    # generate_response() raises if actually called.
    logger.warning(
        "No Gemini API keys found. Set %s in your environment or .env file.",
        " / ".join(KEY_ENV_VARS),
    )

# Index of the key to try first. Advances past keys that fail, so an exhausted
# key is not re-tried at the front of the chain on every subsequent request.
_sticky_index = 0
_sticky_lock = threading.Lock()


def api_error_code(exc: BaseException) -> int | None:
    """
    Find the HTTP status of the underlying Gemini error, or None if there isn't one.

    The status cannot be read off the outermost exception. ``langchain-google-genai``
    catches ``google.genai`` ``ClientError`` and re-raises it as
    ``ChatGoogleGenerativeAIError``, which is NOT an ``APIError`` subclass - so an
    ``isinstance`` check against the raised type misses every 4xx, including the 429s
    that key rotation exists to handle.

    It does use ``raise ... from e``, so the original error survives on ``__cause__``.
    Walking that chain is what makes classification work regardless of how many
    layers wrap the error.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, APIError):
            code = getattr(current, "code", None)
            if isinstance(code, int):
                return code
        # Only __cause__ (explicit `raise ... from`), never __context__, which would
        # pick up unrelated exceptions that merely happened to be in flight.
        current = current.__cause__
    return None


def _is_retryable(exc: BaseException) -> bool:
    """True if a different API key might succeed where this one failed."""
    code = api_error_code(exc)
    if code is None:
        return False
    if code in RETRYABLE_STATUS:
        return True
    return 500 <= code < 600


def _advance_sticky(failed_index: int) -> None:
    """Move the starting point past a key that just failed."""
    global _sticky_index
    with _sticky_lock:
        # Only advance if this key is still the designated start. Without this
        # guard, two concurrent failures on the same key would each increment and
        # skip a healthy key in between.
        if _sticky_index == failed_index:
            _sticky_index = (failed_index + 1) % len(_clients)


def _wrap(index: int, schema: type | None = None) -> Runnable:
    """
    Adapt one client into a Runnable that classifies its own failures.

    ``with_fallbacks`` matches on exception *type*, but Gemini reports quota and
    malformed-request failures with the same ``ClientError`` type, distinguished
    only by an integer ``.code``. This wrapper does the classification and
    re-raises retryable failures as ``_KeyFailure`` so the fallback chain can be
    scoped to exactly those. Everything else propagates untouched and surfaces
    immediately.
    """
    # with_structured_output is applied per client, before the fallback chain is
    # composed, so every key produces schema-validated output rather than only the
    # first one.
    client = _clients[index]
    runnable = client.with_structured_output(schema) if schema is not None else client

    def _invoke(messages: list[BaseMessage]):
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            # Catch broadly and let api_error_code() decide. Anything that is not a
            # retryable Gemini error - including ordinary bugs like TypeError - is
            # re-raised untouched, so nothing is swallowed by catching wide here.
            if not _is_retryable(exc):
                raise
            logger.warning(
                "API key %d/%d failed (HTTP %s); falling back to next key.",
                index + 1,
                len(_clients),
                api_error_code(exc),
            )
            _advance_sticky(index)
            raise _KeyFailure(f"key {index + 1} failed") from exc

    return RunnableLambda(_invoke)


def _build_chain(schema: type | None = None) -> Runnable:
    """
    Build a fallback chain over all keys, starting at the sticky index.

    Rebuilt per request because the sticky index moves. This is cheap: composing
    Runnables only wires up object references, it does no I/O and constructs no
    clients.
    """
    with _sticky_lock:
        start = _sticky_index

    ordered = [
        _wrap((start + offset) % len(_clients), schema) for offset in range(len(_clients))
    ]
    primary, *fallbacks = ordered

    if not fallbacks:
        return primary
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=(_KeyFailure,))


def generate(messages: list[BaseMessage]) -> str:
    """
    Send a prepared message list to Gemini and return the model's text.

    This is the primary entry point. Callers that hold a real conversation
    (``/chat``) pass genuine ``HumanMessage``/``AIMessage`` turns, so the model
    receives proper role boundaries and candidate text can never occupy the
    assistant role.

    Raises:
        LLMUnavailableError: if no key could serve the request.
    """
    return _invoke_chain(messages).text.strip() or _empty_response()


def generate_structured(messages: list[BaseMessage], schema: type):
    """
    Same as generate(), but the model must return an instance of ``schema``.

    Binding the response to a Pydantic model turns the output shape from a
    request into a contract: a missing field or an invented enum value raises
    instead of reaching the caller as malformed text.
    """
    result = _invoke_chain(messages, schema)
    if result is None:
        raise LLMUnavailableError("Model returned no structured output.")
    return result


def _empty_response() -> str:
    # Usually a safety filter or an exhausted token budget. Empty output would
    # otherwise be written into the transcript as a silent blank turn.
    raise LLMUnavailableError("Model returned an empty response.")


def _invoke_chain(messages: list[BaseMessage], schema: type | None = None):
    """Run the fallback chain and translate failures into LLMUnavailableError."""
    if not _clients:
        raise LLMUnavailableError(
            "No Gemini API keys configured. Set "
            f"{KEY_ENV_VARS[0]} in your environment or .env file."
        )
    if not messages:
        raise ValueError("messages must not be empty")

    try:
        return _build_chain(schema).invoke(messages)
    except _KeyFailure as exc:
        logger.error("All %d Gemini API key(s) failed with retryable status: %s", len(_clients), exc, exc_info=True)
        raise LLMUnavailableError(
            f"All {len(_clients)} API key(s) failed. All are rate limited, "
            "out of quota, or invalid."
        ) from exc
    except Exception as exc:
        code = api_error_code(exc)
        logger.error("Gemini API call failed (HTTP %s): %s", code, exc, exc_info=True)
        if code is None:
            # Not a Gemini API failure at all - a genuine bug. Let it surface with
            # its own traceback rather than disguising it as an outage.
            raise
        # Non-retryable API failure: a malformed request (400) or an unknown model
        # (404). Reported with the status, original exception kept as __cause__.
        raise LLMUnavailableError(
            f"Gemini rejected the request (HTTP {code}): {exc}"
        ) from exc


def generate_response(prompt: str, system_instruction: str | None = None) -> str:
    """
    Convenience wrapper for genuinely single-turn calls.

    Used where the model is not a conversation participant: generating the
    opening question, and grading a finished transcript. Those callers have no
    prior turns to represent, so a system instruction plus one human message is
    the honest shape.
    """
    messages: list[BaseMessage] = []
    if system_instruction:
        messages.append(SystemMessage(content=system_instruction))
    messages.append(HumanMessage(content=prompt))
    return generate(messages)
