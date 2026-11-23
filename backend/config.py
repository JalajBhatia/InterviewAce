"""
Environment-driven configuration.

Everything that differs between a laptop and a deployed host is read here, so
deploying is a matter of setting variables rather than editing code.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repository root, resolved from this file rather than the working directory.
# The static mount used a bare relative path, so the app only imported
# successfully when launched from the repo root - which would have broken the
# moment a container set a different WORKDIR.
BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

# Detect serverless / read-only deployment environments (e.g. Vercel, AWS Lambda)
IS_SERVERLESS = bool(
    os.getenv("VERCEL")
    or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
    or os.getenv("LAMBDA_TASK_ROOT")
)

# On serverless platforms the project root is read-only; checkpoints must live in /tmp
_default_data_dir = (Path(tempfile.gettempdir()) / "interviewace") if IS_SERVERLESS else (BASE_DIR / "data")
DATA_DIR = Path(os.getenv("DATA_DIR") or _default_data_dir)

# Where the checkpoint database lives. This holds in-progress interviews only -
# there is no session history and nothing is retained across visits, so losing
# this file costs at most whatever interviews are running at the time.
CHECKPOINT_PATH = DATA_DIR / "checkpoints.db"

# Hosted MySQL Database configuration (Optional, for persistent cloud session storage)
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT") or 3306)
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
MYSQL_SSL = os.getenv("MYSQL_SSL", "false").lower() in ("true", "1", "yes")
MYSQL_URL = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL")

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS") or 15)

# Upload ceiling. A resume is a handful of pages; anything far larger is either a
# mistake or an attempt to exhaust memory.
MAX_RESUME_BYTES = int(os.getenv("MAX_RESUME_BYTES") or 5 * 1024 * 1024)

# Empty by default: the frontend is same-origin, so no cross-origin access is
# needed and none is granted. Set only if hosting the frontend separately.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in (os.getenv("ALLOWED_ORIGINS") or "").split(",")
    if origin.strip()
]
