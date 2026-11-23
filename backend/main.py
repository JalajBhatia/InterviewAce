from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import uuid
import json
import os

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import logging
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

# Import our modules
from .llm_client import LLMUnavailableError, generate, generate_response, generate_structured
from .prompts import (
    SYSTEM_PROMPT_TEMPLATE,
    ANSWER_EVALUATION_RULES,
    INITIAL_QUESTION_PROMPT,
    FEEDBACK_PROMPT,
    FEEDBACK_SYSTEM_PROMPT
)
from .resume_parser import parse_resume
from .schemas import AnswerEvaluation, InterviewFeedback, PerformanceSummary, ResumeScore, RubricScores
from .graph import interview_graph, thread_config
from .resume_analysis import analyze_resume, infer_target_role
from .resume_review import analyse_claims, write_narrative
from .resume_score import score_resume
from .rubric import compute_performance_summary, sanitize_answer_evaluation
from .config import ALLOWED_ORIGINS, FRONTEND_DIR, MAX_QUESTIONS, MAX_RESUME_BYTES
from .limits import rate_limit_middleware
from . import db
import prompts

logger = logging.getLogger(__name__)

# Initialize MySQL tables if cloud credentials are provided
db.init_mysql_schema()

app = FastAPI(
    title="InterviewAce API",
    description="InterviewAce is an AI-powered mock interview assistant that helps students prepare for internships and campus placements through role-specific questions, structured answer evaluation, personalized feedback, and adaptive follow-up questions."
)

@app.exception_handler(LLMUnavailableError)
async def llm_unavailable_handler(request: Request, exc: LLMUnavailableError):
    logger.error("LLM error on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=503,
        content={
            "detail": str(exc),
            "code": "LLM_UNAVAILABLE",
            "message": str(exc),
        },
    )

FALLBACK_INITIAL_QUESTIONS: dict[str, dict[str, str]] = {
    "Software Development Intern": {
        "Technical": (
            "Welcome to InterviewAce! I'll be conducting your technical interview for the Software Development Intern role. "
            "To kick off: Could you explain the key differences between an Array and a Linked List, including time complexity for access, insertion, and deletion, and give an example of when you would choose a Linked List?"
        ),
        "HR": (
            "Welcome to InterviewAce! I'll be conducting your HR & Behavioral interview for the Software Development Intern role. "
            "To begin: Could you introduce yourself, tell me about your background, and share what motivated you to pursue software engineering?"
        ),
    },
    "Frontend Developer": {
        "Technical": (
            "Welcome to your Frontend Developer interview! Let's begin with web architecture: "
            "Could you explain the difference between Client-Side Rendering (CSR) and Server-Side Rendering (SSR), and when you would choose one over the other in a production application?"
        ),
        "HR": (
            "Welcome to InterviewAce! Tell me about a time when you received difficult feedback on a user interface you built, and how you responded to improve the user experience."
        ),
    },
    "Backend Developer": {
        "Technical": (
            "Welcome to your Backend Developer interview! Let's start with database fundamentals: "
            "Could you explain the difference between SQL and NoSQL databases, and walk me through the ACID properties in relational database systems?"
        ),
        "HR": (
            "Welcome to InterviewAce! Could you describe a challenging technical bug or outage you encountered in a project, and how you communicated with your team while solving it?"
        ),
    },
    "Data Analyst": {
        "Technical": (
            "Welcome to your Data Analyst interview! Let's start with SQL and data exploration: "
            "Could you explain the difference between WHERE and HAVING clauses in SQL, and describe a scenario where you would use window functions?"
        ),
        "HR": (
            "Welcome to InterviewAce! Tell me about a situation where you had to present complex data insights to non-technical stakeholders. How did you ensure your message was clear?"
        ),
    },
    "Machine Learning Intern": {
        "Technical": (
            "Welcome to your Machine Learning Intern interview! To start our discussion: "
            "Can you explain the bias-variance tradeoff in machine learning, and describe two practical strategies you use to prevent overfitting when training models?"
        ),
        "HR": (
            "Welcome to InterviewAce! What inspired you to specialize in Machine Learning, and what is one recent AI research paper or practical project you found particularly fascinating?"
        ),
    },
    "HR Interview": {
        "HR": (
            "Welcome to your InterviewAce HR & Behavioral interview! Let's start with our first question: "
            "Tell me about yourself, what drew you to this opportunity, and how your academic or project experiences have prepared you for this role."
        ),
        "Technical": (
            "Welcome! In this behavioral practice round: Walk me through a time you had a disagreement with a teammate on a project. How did you handle the situation and what was the outcome?"
        ),
    },
}


def get_fallback_initial_question(role: str, interview_type: str = "Technical", difficulty: str = "Medium") -> str:
    role_dict = FALLBACK_INITIAL_QUESTIONS.get(role)
    if role_dict:
        msg = role_dict.get(interview_type) or role_dict.get("Technical") or role_dict.get("HR")
        if msg:
            return msg
    if interview_type == "HR":
        return f"Welcome to your InterviewAce mock interview for {role}! To begin: Please tell me about yourself, your background, and why you are interested in this position."
    return f"Welcome to your InterviewAce mock interview for {role}! To begin our discussion: Could you introduce yourself and walk me through a key technical project you have developed?"

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Rate limiting lives here rather than in each endpoint's signature. It is a
# property of the transport, not of the business logic - which also means the
# tests, which call the endpoint functions directly, exercise the logic without
# tripping over a limiter that has nothing to do with what they are asserting.
app.middleware("http")(rate_limit_middleware)

# The frontend is served by this app at /app, so nothing is cross-origin and this
# middleware normally does nothing. It previously combined allow_origins=["*"]
# with allow_credentials=True, which browsers reject outright as invalid - and the
# wildcard existed only because the frontend hardcoded localhost:8000 as its API
# host. With that fixed, no origin is granted access by default; ALLOWED_ORIGINS
# opts in anyone genuinely hosting the frontend elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=bool(ALLOWED_ORIGINS),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Resolved from the package location, not the working directory, so the app can
# be launched from anywhere (and from a container with any WORKDIR).
app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

class StartSessionRequest(BaseModel):
    role: str = "Software Development Intern"
    experience_level: str = "Junior"
    interview_type: str = "Technical"
    difficulty: str = "Medium"
    question_count: int = 5
    prompt_version: str = "v4"
    resume_text: Optional[str] = None
    candidate_name: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    paste_detected: bool = False

class ChatResponse(BaseModel):
    agent_message: str
    is_interview_over: bool = False
    is_completed: bool = False
    evaluation: Optional[AnswerEvaluation] = None
    action: Optional[str] = None
    questions_asked: int = 0
    max_questions: int = 5
    session_total_score: Optional[int] = None
    max_possible_score: Optional[int] = None
    in_intro_stage: bool = False

class DetectRoleRequest(BaseModel):
    # Exactly one is expected. resume_text comes from /upload_resume; goal_statement
    # is the candidate typing what they want to practise for.
    resume_text: Optional[str] = None
    goal_statement: Optional[str] = None


class ScoreResumeRequest(BaseModel):
    resume_text: str
    role: str
    # A real posting is the better target - it is what an applicant tracking
    # system actually matches against - so it wins when supplied.
    job_description: Optional[str] = None
    # When present, resume claims are cross-referenced against what the candidate
    # actually said. This is the part a resume scanner cannot do.
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    # Only the session id is needed. This previously reused ChatRequest, which
    # forced callers to send a user_message the endpoint never read. Pydantic
    # ignores unknown fields, so the frontend's existing payload still validates.
    session_id: str


class FeedbackResponse(BaseModel):
    feedback: InterviewFeedback


def transcript_as_json(history: List[dict]) -> str:
    """
    Serialise the transcript for grading.

    Deliberately JSON rather than a hand-rolled delimiter. json.dumps escapes
    quotes, newlines and control characters correctly, so candidate text cannot
    break out of its string and forge a turn boundary. Inventing a delimiter
    (<turn>...</turn>) would mean owning an escaping scheme forever.

    Unlike /chat this is NOT passed as chat messages: here the model is grading a
    finished transcript from the outside, not participating in it. Handing it real
    AIMessage turns would tell it that it is in the conversation, then ask it to
    assess itself.
    """
    turns = [
        {
            "speaker": "interviewer" if turn["role"] == "model" else "candidate",
            "text": turn["content"],
        }
        for turn in history
    ]
    return json.dumps(turns, indent=2, ensure_ascii=False)


def load_state(session_id: str) -> dict | None:
    """
    Read a session's state from the checkpoint, or None if there is no such session.

    The checkpoint is the only store. There is deliberately no session table and
    no history: nothing identifies a user, and nothing is retained across visits.
    An interview exists only while it is being taken - the checkpoint is there so
    a restart or a dropped connection does not destroy one in progress.
    """
    snapshot = interview_graph.get_state(thread_config(session_id))
    values = snapshot.values or {}
    return values or None


def transcript_from_checkpoint(session_id: str) -> List[dict]:
    """
    Read the conversation back out of the checkpointer.

    Transcripts are stored exactly once, as graph state. The sessions table holds
    only queryable metadata, so there is no second copy to keep in sync.
    """
    snapshot = interview_graph.get_state(thread_config(session_id))
    messages = (snapshot.values or {}).get("messages", [])
    return [
        {
            "role": "model" if isinstance(message, AIMessage) else "user",
            "content": message.text,
        }
        for message in messages
    ]


def history_to_messages(history: List[dict]) -> List[BaseMessage]:
    """
    Convert stored session turns into LangChain messages.

    Sessions record turns as {"role": "model"|"user", "content": str}. Mapping
    them to AIMessage/HumanMessage gives the model real role boundaries instead
    of a flattened "role: content" string. This is what makes turn forgery
    impossible: candidate text is always carried in a HumanMessage, so it cannot
    occupy the assistant role no matter what it contains.
    """
    messages: List[BaseMessage] = []
    for entry in history:
        content = entry["content"]
        if entry["role"] == "model":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))
    return messages

@app.get("/app")
def redirect_app_root():
    return RedirectResponse(url="/app/")

@app.get("/api/index.py", include_in_schema=False)
@app.get("/api/index", include_in_schema=False)
@app.get("/")
@app.get("/ready")
@app.get("/ready/", include_in_schema=False)
@app.get("/health")
@app.get("/health/", include_in_schema=False)
def health_check(request: Request = None):
    """
    Readiness, not just liveness.

    A process that is up but has no API key configured will accept traffic and
    fail on the first real request. Reporting that here means a platform health
    check catches a misconfigured deployment instead of the first visitor doing it.
    """
    if request and "text/html" in request.headers.get("accept", "").split(",")[0]:
        return RedirectResponse(url="/app/")

    from .llm_client import API_KEYS, MODEL_NAME

    ready = bool(API_KEYS)
    return {
        "status": "ok" if ready else "degraded",
        "message": "InterviewAce Backend is running",
        "app": "InterviewAce",
        "model": MODEL_NAME,
        "api_keys_configured": len(API_KEYS),
        "ready": ready,
        "database": db.check_mysql_health(),
    }


@app.get("/db/status", tags=["system"])
@app.get("/db/status/", tags=["system"], include_in_schema=False)
def db_status():
    """Return MySQL database health and connectivity status."""
    return db.check_mysql_health()


@app.get("/prompt_versions")
@app.get("/prompt_versions/", include_in_schema=False)
def list_prompt_versions():
    """Return available prompt versions for developer and evaluation mode."""
    return {
        "versions": [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
            }
            for k, v in prompts.PROMPT_VERSIONS.items()
        ],
        "default": prompts.DEFAULT_PROMPT_VERSION,
    }

# --- Resume Handling ---
@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    # Read in bounded chunks. `await file.read()` pulled the whole upload into
    # memory with no ceiling, so a large file was a trivial way to exhaust it,
    # and pypdf on hostile input can amplify that further.
    chunks, total = [], 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > MAX_RESUME_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Resume too large. Maximum {MAX_RESUME_BYTES // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)
    content = b"".join(chunks)
    parsed = parse_resume(content, file.filename)
    analysis = analyze_resume(parsed.text, parsed.warnings)
    return {
        "resume_text": analysis.text,
        "usable": analysis.is_usable,
        "skills": analysis.skills,
        "bullet_count": len(analysis.bullets),
        # Surfaced so a failed extraction is visible instead of looking like success.
        "warnings": analysis.warnings,
    }


@app.post("/detect_role")
def detect_role(request: DetectRoleRequest):
    """
    Infer the target role from a resume or a free-text goal, for the user to confirm.

    Deliberately its own endpoint rather than folded into /start_session: the
    result is a suggestion the user accepts or overrides, so it has to come back
    before the interview begins.
    """
    source = (request.resume_text or "").strip()
    kind = "resume"
    if not source:
        source = (request.goal_statement or "").strip()
        kind = "statement"

    if len(source) < 20:
        raise HTTPException(
            status_code=400,
            detail="Provide a resume or a sentence describing the role you want to practise for.",
        )

    detection = infer_target_role(source, kind)
    return {
        "role": detection.role,
        "confidence": detection.confidence,
        "reasoning": detection.reasoning,
        "skills_found": detection.skills_found,
        "suggested_experience_level": detection.suggested_experience_level,
    }

MANDATORY_INTRO_QUESTION = (
    "Hello, and welcome! To get started, could you please tell me about yourself and your background?"
)

PASTED_WARNING = (
    "Pasted answer detected. Please answer in your own words to receive a score."
)


@app.post("/start_session")
def start_session(request: StartSessionRequest):
    session_id = str(uuid.uuid4())
    resume_context = request.resume_text or ""
    prompt_ver = request.prompt_version or prompts.DEFAULT_PROMPT_VERSION
    max_q = max(1, min(15, request.question_count or MAX_QUESTIONS))

    initial_message = MANDATORY_INTRO_QUESTION

    # Seed the checkpoint without running any node. update_state writes state
    # directly, which is what we want here: the greeting was produced outside the
    # graph, and invoking would run classify/conduct and ask a second question.
    interview_graph.update_state(
        thread_config(session_id),
        {
            "role": request.role,
            "interview_type": request.interview_type,
            "difficulty": request.difficulty,
            "experience_level": request.experience_level,
            "prompt_version": prompt_ver,
            "resume_text": resume_context,
            "messages": [AIMessage(content=initial_message)],
            "questions_asked": 0,
            "max_questions": max_q,
            "persona": "normal",
            "end_requested": False,
            "is_over": False,
            "last_action": "intro",
            "evaluations": [],
            "last_evaluation": None,
            "feedback": None,
            "current_question_follow_ups": 0,
            "in_intro_stage": True,
            "in_closing_stage": False,
            "paste_detected": False,
        },
    )

    db.save_session(
        session_id=session_id,
        role=request.role,
        experience_level=request.experience_level,
        difficulty=request.difficulty,
        question_count=max_q,
    )
    db.add_transcript_turn(session_id, 0, "model", initial_message)

    return {
        "session_id": session_id,
        "initial_message": initial_message,
        "role": request.role,
        "interview_type": request.interview_type,
        "difficulty": request.difficulty,
        "max_questions": max_q,
        "prompt_version": prompt_ver,
        "in_intro_stage": True,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    state = load_state(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.get("is_over"):
        evaluations = state.get("evaluations") or []
        max_q = state.get("max_questions", 5)
        summary = compute_performance_summary(evaluations, max_questions=max_q)
        return ChatResponse(
            agent_message="The interview is already over. Please request feedback.",
            is_interview_over=True,
            is_completed=True,
            action="close",
            questions_asked=min(state.get("questions_asked", 0), max_q),
            max_questions=max_q,
            session_total_score=summary.total_marks_obtained,
            max_possible_score=max_q * 20,
        )

    cleaned_message = (request.user_message or "").strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=400,
            detail="Cannot submit an empty answer. Please provide a response.",
        )

    # Only the new message is sent. The checkpointer restores the transcript,
    # question count and role context from the previous turn, and the
    # add_messages reducer appends rather than replacing.
    result = interview_graph.invoke(
        {
            "messages": [HumanMessage(content=cleaned_message)],
            "paste_detected": bool(request.paste_detected),
        },
        thread_config(request.session_id),
    )

    last_eval = result.get("last_evaluation")
    parsed_eval = None
    is_completed = bool(result.get("is_over") or result.get("last_action") == "close")

    if last_eval and not is_completed:
        try:
            parsed_eval = AnswerEvaluation(**last_eval)
        except Exception:
            parsed_eval = None

    # Backend enforcement of paste penalty: override after LLM evaluation step
    if request.paste_detected and parsed_eval is not None and not is_completed:
        parsed_eval.rubric = RubricScores(
            correctness=0,
            relevance=0,
            clarity=0,
            completeness=0,
            total_score=0,
            percentage=0.0,
        )
        parsed_eval.strengths = ["No credit: pasted content detected."]
        parsed_eval.areas_for_improvement = [PASTED_WARNING]
        parsed_eval.paste_detected = True

    evaluations = list(result.get("evaluations") or [])
    if request.paste_detected and evaluations and parsed_eval is not None and not is_completed:
        evaluations[-1] = parsed_eval.model_dump()
        interview_graph.update_state(
            thread_config(request.session_id),
            {"evaluations": evaluations, "last_evaluation": parsed_eval.model_dump()},
        )

    max_q = result.get("max_questions", 5)
    valid_technical_evals = []
    for e in evaluations:
        q_num = getattr(e, "question_number", None) if not isinstance(e, dict) else e.get("question_number")
        if q_num is not None:
            try:
                val = int(q_num)
                if val <= 0 or val > max_q:
                    continue
            except (ValueError, TypeError):
                pass
        valid_technical_evals.append(e)
    summary = compute_performance_summary(valid_technical_evals, max_questions=max_q)

    agent_msg = result["messages"][-1].text
    q_asked = min(result.get("questions_asked", 0), max_q)

    turn_idx = len(result["messages"])
    db.add_transcript_turn(request.session_id, turn_idx - 1, "user", request.user_message)
    db.add_transcript_turn(request.session_id, turn_idx, "model", agent_msg)
    db.update_session(
        session_id=request.session_id,
        questions_asked=q_asked,
        session_total_score=summary.total_marks_obtained,
        max_possible_score=max_q * 20,
        is_over=is_completed,
    )

    return ChatResponse(
        agent_message=agent_msg,
        is_interview_over=is_completed,
        is_completed=is_completed,
        evaluation=parsed_eval,
        action=result.get("last_action"),
        questions_asked=q_asked,
        max_questions=max_q,
        session_total_score=summary.total_marks_obtained,
        max_possible_score=max_q * 20,
        in_intro_stage=bool(result.get("in_intro_stage", False)),
    )


@app.post("/feedback", response_model=FeedbackResponse)
def get_feedback(request: FeedbackRequest):
    state = load_state(request.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Cached in the session's own checkpoint, so it lives and dies with the
    # interview. Without this every call is a fresh LLM request on an
    # unauthenticated endpoint, and the frontend double-fires it when End
    # Interview is clicked during the auto-end timer.
    evaluations = state.get("evaluations") or []
    max_q = state.get("max_questions", 5)
    valid_technical_evals = []
    for e in evaluations:
        q_num = getattr(e, "question_number", None) if not isinstance(e, dict) else e.get("question_number")
        if q_num is not None:
            try:
                val = int(q_num)
                if val <= 0 or val > max_q:
                    continue
            except (ValueError, TypeError):
                pass
        valid_technical_evals.append(e)
    perf_summary = compute_performance_summary(valid_technical_evals, max_questions=max_q)

    if state.get("feedback"):
        cached = InterviewFeedback(**state["feedback"])
        if not cached.performance_summary and perf_summary:
            cached.performance_summary = perf_summary
        return FeedbackResponse(feedback=cached)

    history = transcript_from_checkpoint(request.session_id)
    answer_count = sum(1 for turn in history if turn["role"] == "user")
    if answer_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate feedback: the candidate has not answered any questions.",
        )

    system_instruction = FEEDBACK_SYSTEM_PROMPT.format(
        role=state["role"],
        experience_level=state["experience_level"],
        answer_count=answer_count,
    )
    prompt = FEEDBACK_PROMPT.format(transcript_json=transcript_as_json(history))

    feedback = generate_structured(
        [SystemMessage(content=system_instruction), HumanMessage(content=prompt)],
        InterviewFeedback,
    )

    # Attach the actual mathematical rubric summary
    feedback.performance_summary = perf_summary

    interview_graph.update_state(
        thread_config(request.session_id),
        {"feedback": feedback.model_dump(), "is_over": True},
    )
    db.update_session(
        session_id=request.session_id,
        is_over=True,
        feedback=feedback.model_dump(),
    )
    return FeedbackResponse(feedback=feedback)


@app.post("/score_resume", response_model=ResumeScore)
def score_resume_endpoint(request: ScoreResumeRequest):
    """
    Score a resume, optionally against a job description and an interview.

    The number is computed entirely by deterministic rules before any model is
    called, so the same resume always scores the same and every point traces to a
    named finding. The LLM only writes the narrative and, when a transcript
    exists, the claim cross-reference - neither can alter the score.
    """
    analysis = analyze_resume(request.resume_text)
    if not analysis.text.strip():
        raise HTTPException(
            status_code=400, detail="No resume text to score."
        )

    score = score_resume(analysis, request.role, request.job_description)

    if request.session_id:
        history = transcript_from_checkpoint(request.session_id)
        # Needs actual answers to compare against; an unanswered interview has no
        # evidence to hold the resume up to.
        if any(turn["role"] == "user" for turn in history):
            score.claim_analysis = analyse_claims(analysis.text, history)

    score.narrative = write_narrative(score)
    return score
app.mount(
    "/",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)