"""
MySQL database persistence layer for InterviewAce.

Provides optional persistent cloud storage for interview sessions, scores,
and transcripts when hosted MySQL credentials (e.g., PlanetScale, AWS RDS,
Aiven, TiDB Cloud, Railway) are configured via environment variables.

If MySQL environment variables are not configured, InterviewAce smoothly defaults
to SQLite checkpointing without errors.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
from typing import Any, Dict, List, Optional

from .config import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_SSL,
    MYSQL_URL,
    MYSQL_USER,
)

logger = logging.getLogger(__name__)

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    pymysql = None


def get_mysql_config() -> Optional[Dict[str, Any]]:
    """
    Parse and validate MySQL connection configuration.
    Supports individual MYSQL_* variables or full MYSQL_URL / DATABASE_URL.
    Reads dynamically from environment, falling back to config defaults.
    """
    mysql_url = os.getenv("MYSQL_URL") or os.getenv("DATABASE_URL") or MYSQL_URL
    mysql_ssl = (
        os.getenv("MYSQL_SSL", "false").lower() in ("true", "1", "yes")
        if "MYSQL_SSL" in os.environ
        else MYSQL_SSL
    )

    if mysql_url:
        try:
            parsed = urllib.parse.urlparse(mysql_url)
            return {
                "host": parsed.hostname or "localhost",
                "port": parsed.port or 3306,
                "user": parsed.username or "root",
                "password": parsed.password or "",
                "database": parsed.path.lstrip("/"),
                "ssl": bool(mysql_ssl or "ssl" in parsed.query.lower()),
            }
        except Exception as e:
            logger.warning("Failed to parse MYSQL_URL / DATABASE_URL: %s", e)

    host = os.getenv("MYSQL_HOST") or MYSQL_HOST
    user = os.getenv("MYSQL_USER") or MYSQL_USER
    database = os.getenv("MYSQL_DATABASE") or MYSQL_DATABASE
    port = int(os.getenv("MYSQL_PORT") or MYSQL_PORT or 3306)
    password = os.getenv("MYSQL_PASSWORD") if "MYSQL_PASSWORD" in os.environ else (MYSQL_PASSWORD or "")

    if host and user and database:
        return {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
            "ssl": mysql_ssl,
        }

    return None


def is_mysql_configured() -> bool:
    """Return True if hosted MySQL database credentials are configured."""
    return get_mysql_config() is not None


def get_connection():
    """Create and return a new pymysql connection if configured."""
    if pymysql is None:
        logger.warning("pymysql package is not installed; MySQL support disabled.")
        return None

    cfg = get_mysql_config()
    if not cfg:
        return None

    ssl_args = None
    if cfg["ssl"]:
        ssl_args = {"ssl": True}

    return pymysql.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
        charset="utf8mb4",
        connect_timeout=5,
        **ssl_args if ssl_args else {},
    )


def init_mysql_schema() -> bool:
    """
    Idempotently initialize required database tables.
    Returns True if successful, False if MySQL is not configured or fails.
    """
    if not is_mysql_configured():
        return False

    try:
        conn = get_connection()
        if not conn:
            return False
        with conn:
            with conn.cursor() as cursor:
                # 1. Sessions table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interview_sessions (
                        session_id VARCHAR(64) PRIMARY KEY,
                        role VARCHAR(128) NOT NULL,
                        experience_level VARCHAR(64) DEFAULT 'Junior',
                        difficulty VARCHAR(64) DEFAULT 'Medium',
                        question_count INT DEFAULT 5,
                        questions_asked INT DEFAULT 0,
                        session_total_score INT DEFAULT 0,
                        max_possible_score INT DEFAULT 100,
                        is_over BOOLEAN DEFAULT FALSE,
                        feedback_json LONGTEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        INDEX idx_created (created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 2. Transcripts table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interview_transcripts (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(64) NOT NULL,
                        turn_index INT DEFAULT 0,
                        speaker VARCHAR(32) NOT NULL,
                        content LONGTEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_session_turn (session_id, turn_index)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
        logger.info("Hosted MySQL database schema initialized successfully.")
        return True
    except Exception as e:
        logger.error("Failed to initialize MySQL schema: %s", e)
        return False


def save_session(
    session_id: str,
    role: str,
    experience_level: str = "Junior",
    difficulty: str = "Medium",
    question_count: int = 5,
) -> bool:
    """Insert a new interview session record into MySQL."""
    if not is_mysql_configured():
        return False

    try:
        conn = get_connection()
        if not conn:
            return False
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO interview_sessions (
                        session_id, role, experience_level, difficulty, question_count
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        role = VALUES(role),
                        experience_level = VALUES(experience_level),
                        difficulty = VALUES(difficulty),
                        question_count = VALUES(question_count)
                    """,
                    (session_id, role, experience_level, difficulty, question_count),
                )
        return True
    except Exception as e:
        logger.warning("Could not persist session %s to MySQL: %s", session_id, e)
        return False


def update_session(
    session_id: str,
    questions_asked: Optional[int] = None,
    session_total_score: Optional[int] = None,
    max_possible_score: Optional[int] = None,
    is_over: Optional[bool] = None,
    feedback: Optional[dict] = None,
) -> bool:
    """Update an existing session's progress or completion state in MySQL."""
    if not is_mysql_configured():
        return False

    updates = []
    params = []
    if questions_asked is not None:
        updates.append("questions_asked = %s")
        params.append(questions_asked)
    if session_total_score is not None:
        updates.append("session_total_score = %s")
        params.append(session_total_score)
    if max_possible_score is not None:
        updates.append("max_possible_score = %s")
        params.append(max_possible_score)
    if is_over is not None:
        updates.append("is_over = %s")
        params.append(bool(is_over))
    if feedback is not None:
        updates.append("feedback_json = %s")
        params.append(json.dumps(feedback, ensure_ascii=False))

    if not updates:
        return True

    params.append(session_id)
    try:
        conn = get_connection()
        if not conn:
            return False
        with conn:
            with conn.cursor() as cursor:
                query = f"UPDATE interview_sessions SET {', '.join(updates)} WHERE session_id = %s"
                cursor.execute(query, tuple(params))
        return True
    except Exception as e:
        logger.warning("Could not update session %s in MySQL: %s", session_id, e)
        return False


def add_transcript_turn(
    session_id: str,
    turn_index: int,
    speaker: str,
    content: str,
) -> bool:
    """Persist an individual message turn to MySQL."""
    if not is_mysql_configured():
        return False

    try:
        conn = get_connection()
        if not conn:
            return False
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO interview_transcripts (session_id, turn_index, speaker, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (session_id, turn_index, speaker, content),
                )
        return True
    except Exception as e:
        logger.warning("Could not persist turn for session %s to MySQL: %s", session_id, e)
        return False


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve session record from MySQL."""
    if not is_mysql_configured():
        return None

    try:
        conn = get_connection()
        if not conn:
            return None
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM interview_sessions WHERE session_id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row and row.get("feedback_json"):
                    try:
                        row["feedback"] = json.loads(row["feedback_json"])
                    except Exception:
                        pass
                return row
    except Exception as e:
        logger.warning("Could not read session %s from MySQL: %s", session_id, e)
        return None


def check_mysql_health() -> Dict[str, Any]:
    """Check connectivity to the configured MySQL database without exposing secrets."""
    cfg = get_mysql_config()
    if not cfg:
        return {
            "configured": False,
            "connected": False,
            "type": "sqlite",
            "message": "Hosted MySQL not configured. Using SQLite checkpointer.",
        }

    try:
        conn = get_connection()
        if conn:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1 AS ok")
            return {
                "configured": True,
                "connected": True,
                "type": "mysql",
                "host": cfg["host"],
                "database": cfg["database"],
                "port": cfg["port"],
            }
        return {
            "configured": True,
            "connected": False,
            "type": "mysql",
            "error": "Failed to create connection object.",
        }
    except Exception as e:
        return {
            "configured": True,
            "connected": False,
            "type": "mysql",
            "error": str(e),
        }
