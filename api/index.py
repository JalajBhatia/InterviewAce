"""
Vercel Serverless Function Entrypoint for InterviewAce.
Exposes the FastAPI application instance at backend.main:app.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is placed on sys.path so 'backend' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.main import app

# Vercel looks for 'app' as the WSGI/ASGI handler
__all__ = ["app"]
