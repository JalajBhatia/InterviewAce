"""
Vercel Serverless Function Entrypoint for InterviewAce.
Exposes the FastAPI application instance at backend.main:app,
wrapped with ASGI middleware to normalize request paths when Vercel serverless
rewrites route requests to api/index.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is placed on sys.path so 'backend' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app as fastapi_app


class VercelPathNormalizer:
    """
    ASGI middleware wrapper that ensures requests rewritten by Vercel to
    api/index.py preserve their true client request path (e.g. /health, /app/, /db/status, /).

    Without this normalizer, Vercel's rewrite rules cause Starlette/FastAPI
    to see scope['path'] == '/api/index.py' (or '/api/index'), which fails to match
    any FastAPI route and yields a 404 {"detail": "Not Found"}.
    """

    def __init__(self, asgi_app):
        self.asgi_app = asgi_app

    def __getattr__(self, name: str):
        return getattr(self.asgi_app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            method = scope.get("method", "UNKNOWN")
            incoming_path = scope.get("path", "")

            # Vercel and reverse proxies pass the original client path in standard headers
            h_fwd = headers.get(b"x-forwarded-uri", b"").decode("utf-8", errors="ignore")
            h_orig = headers.get(b"x-original-url", b"").decode("utf-8", errors="ignore")
            h_rewrite = headers.get(b"x-rewrite-url", b"").decode("utf-8", errors="ignore")
            h_matched = headers.get(b"x-matched-path", b"").decode("utf-8", errors="ignore")

            matched = h_fwd or h_orig or h_rewrite or h_matched

            target_path = incoming_path

            # If an authoritative forwarded path was provided by the proxy, use it
            if matched:
                path_part, _, qs_part = matched.partition("?")
                target_path = path_part
                if qs_part and not scope.get("query_string"):
                    scope["query_string"] = qs_part.encode("utf-8")

            # Handle /api/ prefix or serverless entrypoint rewrites
            if target_path in ("/api/index.py", "/api/index", "/api", ""):
                target_path = "/"
            elif target_path.startswith("/api/index.py/"):
                target_path = target_path[len("/api/index.py"):]
            elif target_path.startswith("/api/index/"):
                target_path = target_path[len("/api/index"):]
            elif target_path.startswith("/api/"):
                # If path was sent with /api/ prefix (e.g. /api/health -> /health)
                stripped = target_path[4:]
                if stripped in ("/index.py", "/index", ""):
                    target_path = "/"
                elif stripped.startswith("/"):
                    target_path = stripped

            # Guarantee leading slash
            if not target_path.startswith("/"):
                target_path = "/" + target_path

            # Temporary diagnostic logging (no sensitive data logged)
            print(
                f"[VercelPathNormalizer] method={method} incoming_path='{incoming_path}' "
                f"x-forwarded-uri='{h_fwd}' x-original-url='{h_orig}' "
                f"x-rewrite-url='{h_rewrite}' x-matched-path='{h_matched}' "
                f"target_path='{target_path}'",
                flush=True,
            )

            scope["path"] = target_path
            scope["raw_path"] = target_path.encode("utf-8")

            async def debug_send(message):
                if message["type"] == "http.response.start":
                    resp_headers = list(message.get("headers", []))
                    resp_headers.append(
                        (b"x-debug-incoming-path", incoming_path.encode("utf-8"))
                    )
                    resp_headers.append(
                        (b"x-debug-target-path", target_path.encode("utf-8"))
                    )
                    message = {**message, "headers": resp_headers}
                await send(message)

            await self.asgi_app(scope, receive, debug_send)
        else:
            await self.asgi_app(scope, receive, send)

app = VercelPathNormalizer(fastapi_app)

# Vercel looks for 'app' as the WSGI/ASGI handler
__all__ = ["app", "fastapi_app", "VercelPathNormalizer"]
