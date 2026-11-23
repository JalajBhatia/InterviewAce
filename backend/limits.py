"""
Request limits.

Every expensive endpoint here costs an LLM call against a finite quota, and none
of them require authentication. Without a ceiling, a single loop over
/start_session drains the day's budget - and unlike a normal API, the damage is
not just load: it is spend, and it silently breaks the app for real users.

In-process and per-IP, which is the honest scope: one server, one bucket. Behind
multiple workers each would keep its own counter, so the effective limit is
per-worker. A shared store (Redis) is the answer at that point; a token bucket in
memory is the right amount of machinery for a single-instance deployment, and
saying so is better than pretending otherwise.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse

# Per client, per window. Generous enough that a real interview never notices:
# a session is one /start_session, a handful of /chat, one /feedback.
WINDOW_SECONDS = 60
# Keyed by path. Every one of these costs an LLM call.
LIMITS: dict[str, int] = {
    "/start_session": 5,    # starting five interviews a minute is not human
    "/chat": 30,            # a turn every two seconds is already fast speech
    "/feedback": 5,
    "/detect_role": 10,
    "/upload_resume": 10,
    "/score_resume": 5,
}

_hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def client_key(request: Request) -> str:
    """
    Identify the caller.

    X-Forwarded-For is trusted only for its first entry, and only because this is
    intended to sit behind a platform proxy that sets it. A header a client can
    forge is not an identity, so this limits accidental and casual abuse, not a
    determined attacker - that needs authentication.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check(client: str, path: str) -> int | None:
    """Record a hit. Returns seconds to wait if over the limit, else None."""
    limit = LIMITS.get(path)
    if limit is None:
        return None

    key = (client, path)
    now = time.monotonic()
    cutoff = now - WINDOW_SECONDS

    with _lock:
        timestamps = _hits[key]
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()
        if len(timestamps) >= limit:
            return int(WINDOW_SECONDS - (now - timestamps[0])) + 1
        timestamps.append(now)
    return None


async def rate_limit_middleware(request: Request, call_next):
    """
    Applied at the transport layer rather than in endpoint signatures.

    Rate limiting is a property of who is calling over HTTP, not of what the
    interview logic does - so threading a Request through every handler would put
    transport concerns into business functions, and force every unit test to
    fabricate a Request to call code that has nothing to do with limits.
    """
    retry_after = check(client_key(request), request.url.path)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Too many requests. Try again in {retry_after}s."},
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


def reset() -> None:
    """Clear all buckets. Tests only."""
    with _lock:
        _hits.clear()
