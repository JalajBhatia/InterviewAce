# InterviewAce — AI-Powered Mock Interview & Personalized Feedback Assistant

> **Academic Prompt Engineering Mini-Project**  
> **Problem Track:** Problem C10 — Mock Interview Question and Feedback Tool  
> **Target Audience:** University students preparing for campus recruitment drives, tech internships, and HR rounds.

---

## 📌 Project Overview

**InterviewAce** is an AI-powered mock interview assistant that helps students prepare for internships and campus placements through role-specific questions, structured answer evaluation, personalized feedback, and adaptive follow-up questions.

Built upon **FastAPI**, **LangGraph**, and **Google Gemini**, InterviewAce evaluates student responses using a **fixed 4-criteria rubric** (scored out of 20 marks), calculates real-time criterion-wise metrics, generates placement-ready exemplar answers, and conducts comprehensive performance assessments.

### Milestone 1 Requirements Coverage

| # | Milestone 1 Requirement | Implementation in InterviewAce |
|---|---|---|
| **1** | Generate questions based on selected job role | Supports **Software Development Intern**, **Frontend Developer**, **Backend Developer**, **Data Analyst**, **Machine Learning Intern**, **HR Interview**, and 15+ industry roles with tailored question banks. |
| **2** | Support technical and HR interviews | Dedicated **Technical Interview Track** (DSA, coding, system design) and **HR / Behavioral Track** (STAR Method, culture fit, conflict resolution). |
| **3** | Ask one question at a time | Strict single-question state machine; counts questions accurately rather than message turns. |
| **4** | Evaluate student answers using a fixed rubric | Rigorous 4-criteria rubric: **Correctness (0–5)**, **Relevance (0–5)**, **Clarity (0–5)**, and **Completeness (0–5)**. |
| **5** | Provide criterion-wise scores | Every answer displays score breakdowns for all four individual criteria. |
| **6** | Explain strengths and weaknesses | Specific bulleted strengths and actionable areas for improvement citing candidate answers. |
| **7** | Suggest improved answers | Generates high-scoring exemplar model answers showing placement candidates how to structure optimal responses. |
| **8** | Ask relevant follow-up questions | Adaptive turn actions (`advance`, `follow_up`, `rephrase`, `redirect`) probing student gaps or edge cases before moving on. |
| **9** | Generate final performance summary | Aggregates actual session marks, percentage, average score, strongest/weakest criteria, and placement recommendations. |
| **10** | Demonstrate prompt engineering | Systematic versioning (**V1 Baseline**, **V2 Structured**, **V3 Improved**, **V4 Final**) and automated benchmark evaluation across 10 test cases. |

---

## 🔬 How this Project Demonstrates Prompt Engineering

This project is not merely an ordinary web application calling an LLM endpoint. It is an experimental and production-grade demonstration of **applied prompt engineering** in a multi-turn evaluation system:

1. **Systematic Prompt Iteration & Architecture (`prompts/`):**
   - Maintained in dedicated versioned modules (`v1_baseline.py`, `v2_structured.py`, `v3_improved.py`, `v4_final.py`).
   - Each prompt iteration addresses a specific LLM failure mode observed in prior versions.

2. **Mitigating LLM Evaluation Biases:**
   - **Verbosity Bias:** Baseline LLMs reward long, wordy answers even when conceptually hollow or off-topic. V3 and V4 establish explicit constraints ensuring that rambling responses receive low Relevance and Clarity scores.
   - **Hallucination & Sycophancy:** Novice prompts often validate fabricated candidate claims. V4 penalizes ungrounded or exaggerated experience (clamping Correctness to 0–2).
   - **Score Anchoring:** V3 and V4 define rigorous semantic anchors for every score level (5 = Exemplary, 3 = Competent with flaws, 0 = Incorrect or irrelevant).

3. **Schema-Constrained Structured Output:**
   - Evaluator prompts are bound to Pydantic schemas (`RubricScores`, `AnswerEvaluation`, `InterviewTurn`, `InterviewFeedback`).
   - Eliminates brittle string parsing and prevents prompt injection attacks.

4. **Server-Side Integrity Layer:**
   - The backend validates every score, clamps criteria to `[0, 5]`, and deterministically recalculates total marks ($0–20$) and percentages ($\frac{\text{total}}{20} \times 100$) instead of blindly trusting the model's arithmetic.

5. **Empirical Evaluation Benchmark (`evals/`):**
   - 10 standardized candidate test cases (`evals/prompt_test_cases.csv`) covering correct, partially correct, incorrect, very short, irrelevant, shallow, rambling, and fabricated answers.
   - Automated evaluation suite (`evals/evaluate_prompts.py`) measuring format compliance, rubric adherence, discrimination index, and score accuracy.

---

## 📊 Prompt Evolution & Evaluation Benchmark

Below is the empirical comparison of our four prompt iterations evaluated across the 10 standardized test cases:

| Evaluation Metric | V1 (Baseline) | V2 (Structured) | V3 (Improved) | V4 (Final Production) |
|---|---|---|---|---|
| **Format & Schema Compliance** | 0.0% | 100.0% | 100.0% | **100.0%** |
| **Rubric Compliance (4 Criteria)** | 0.0% | 100.0% | 100.0% | **100.0%** |
| **Expected Score Accuracy** | 30.0% | 30.0% | 100.0% | **100.0%** |
| **Discrimination Delta (TC01 vs TC05)** | 6 / 20 | 12 / 20 | 16 / 20 | **16 / 20** |
| **Average Score Across Benchmark** | 16.6 / 20 | 11.7 / 20 | 9.8 / 20 | **9.7 / 20** |

### Iteration Analysis:
- **V1 — Baseline:** Simple interviewer prompt. Suffers from high verbosity bias and lacks criterion breakdowns. Even completely irrelevant answers (TC05: talking about a weather app when asked about ML overfitting) were awarded 14/20.
- **V2 — Structured:** Introduced the 4 criteria, but lacked explicit scoring anchor criteria. Scored rambling and fabricated claims too generously.
- **V3 — Improved:** Added explicit 0–5 anchor definitions, strict output constraints, and anti-hallucination rules. Irrelevant answers correctly plummeted to 0–3/20.
- **V4 — Final Production:** Role-specific technical guidelines, STAR method enforcement for HR, difficulty calibration (Easy, Medium, Hard), high-quality exemplar model answers, and adaptive follow-up questions (`advance`, `follow_up`, `rephrase`, `redirect`).

---

## 📋 The Fixed Evaluation Rubric (20 Marks)

Every answer submitted by a candidate is evaluated in real time across four standardized criteria:

$$\text{Total Score} = \text{Correctness} + \text{Relevance} + \text{Clarity} + \text{Completeness} \quad (\text{out of } 20)$$

$$\text{Overall Percentage} = \left(\frac{\text{Total Score}}{20}\right) \times 100$$

### Rubric Definitions & Anchor Criteria:
1. **Correctness (0–5 Marks):**
   - **5:** Impeccable factual and conceptual accuracy; zero technical misconceptions.
   - **4:** Solid understanding with only minor non-critical omissions.
   - **3:** Broad concept recognized, but technical details contain noticeable flaws.
   - **2:** Substantial inaccuracies or incorrect terminology.
   - **1:** Severe misconceptions throughout.
   - **0:** Flatly wrong, fabricated, or hallucinated claims.

2. **Relevance (0–5 Marks):**
   - **5:** Directly answers the exact question asked without fluff or evasion.
   - **4:** Directly on-topic with slight verbosity.
   - **3:** Answers part of the question; misses the primary focus.
   - **2:** Drifts noticeably off-topic or addresses tangential subjects.
   - **1:** Extremely weak connection to the prompt.
   - **0:** Total evasion, gibberish, or irrelevant topic.

3. **Clarity (0–5 Marks):**
   - **5:** Structured, articulate, concise, and logically organized (STAR method for HR, Input-Approach-Complexity for Tech).
   - **4:** Clear and understandable with slight phrasing awkwardness.
   - **3:** Conversational but rambles or lacks cohesive structure.
   - **2:** Disjointed, fragmented, or difficult to follow.
   - **1:** Highly incoherent.
   - **0:** Single-word answer ("yes", "no", "idk") or unintelligible.

4. **Completeness (0–5 Marks):**
   - **5:** Thorough coverage: addresses core solution, edge cases, trade-offs, and practical reasoning.
   - **4:** Covers all major points thoroughly; minor edge case omitted.
   - **3:** Meets bare minimum baseline; shallow depth.
   - **2:** Incomplete; leaves major expected points unaddressed.
   - **1:** Extremely brief or superficial outline.
   - **0:** Empty or unattempted.

---

## 🛠️ Technology Stack & Architecture

- **Backend:** FastAPI (Python 3.12)
- **State Machine & Persistence:** LangGraph with SQLite checkpointer (`data/checkpoints.db`)
- **LLM Engine:** Google Gemini (`gemini-2.5-flash-lite`) via `langchain-google-genai` and `google-genai` SDK
- **Multi-Key Failover:** Transparent key rotation across `GEMINI_API_KEY`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3` for rate-limit protection
- **Document & Resume Parser:** `pypdf`, `python-docx`
- **ATS Matching Engine:** Lexical BM25 keyword saturation with canonical skill aliasing
- **Frontend:** Vanilla HTML5, modern CSS3 (AI SaaS Dashboard aesthetic), and asynchronous ES6 JavaScript
- **Voice Interaction:** Browser-native Web Speech API (`SpeechRecognition` & `SpeechSynthesis`) with 12s silence watchdog and auto-recovery
- **Testing:** `pytest`, `pytest-asyncio` (179 automated unit tests)

### Session State Diagram:
```
                      START
                        │
                  [ classify ]        Persona + end-intent detection (deterministic)
                        │
                route_after_classify
                   /          \
           [ close ]           [ conduct ]     Structured model inference with Rubric
               │                    │
              END                  END
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (or 3.11+)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or standard `pip`
- A [Gemini API Key](https://aistudio.google.com/app/apikey) from Google AI Studio

### Installation

1. Clone or extract the repository:
   ```bash
   cd interview-practice-partner-039-main
   ```

2. Install dependencies with `uv`:
   ```bash
   uv sync
   ```
   *(Or with pip: `pip install -r requirements.txt`)*

3. Configure Environment Variables:
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY=your_actual_key_here
   ```

4. Run the Application:
   ```bash
   uv run uvicorn backend.main:app --reload
   ```

5. Open in Browser:
   Navigate to **<http://localhost:8000/app/>** (Use Google Chrome or Microsoft Edge for Web Speech API support).

---

## 🧪 Running Tests & Evaluations

### 1. Automated Unit Test Suite
The project includes a comprehensive, fast, offline test suite with 179 tests (using mock clients and local checkpointer):
```bash
uv run pytest
```
*Result: 179 passed in ~1.4 seconds.*

### 2. Prompt Evaluation Suite
Run the prompt benchmark comparing V1, V2, V3, and V4 across the 10 standardized candidate test cases:
```bash
uv run python -m evals.evaluate_prompts
```
*Outputs:*
- `evals/prompt_evaluation_results.json`
- `evals/prompt_evaluation_report.md`

---

## 📁 Project Structure

```
interview-practice-partner-039-main/
├── backend/
│   ├── config.py              # Environment configuration & limits
│   ├── graph.py               # LangGraph state machine & conduct turn handler
│   ├── limits.py              # Rate limiting & upload boundaries
│   ├── llm_client.py          # Gemini LLM client with multi-key failover
│   ├── main.py                # FastAPI routes & API endpoints
│   ├── matching.py            # BM25 lexical resume matching
│   ├── persona_logic.py       # Rule-based candidate persona classifier
│   ├── prompts.py             # Role guidelines & feedback prompts
│   ├── resume_analysis.py     # Resume extraction & target role inference
│   ├── resume_checks.py       # Deterministic resume quality checks
│   ├── resume_parser.py       # PDF/DOCX multi-format parser
│   ├── resume_review.py       # Resume vs transcript claim cross-examiner
│   ├── resume_score.py        # Resume scoring rules
│   ├── roles.py               # Canonical role taxonomy & skill aliases
│   ├── rubric.py              # 4-criteria rubric validation & analytics
│   └── schemas.py             # Pydantic schemas (RubricScores, InterviewFeedback)
├── prompts/                   # Dedicated Prompt Engineering Registry
│   ├── __init__.py            # Prompt version registry & selector
│   ├── v1_baseline.py         # V1 — Baseline prompt
│   ├── v2_structured.py       # V2 — Structured rubric prompt
│   ├── v3_improved.py         # V3 — Anchored criteria & anti-hallucination
│   └── v4_final.py            # V4 — Final production prompt
├── evals/                     # Prompt Evaluation & Behavioral Checks
│   ├── checks.py              # Turn & session behavioral checks
│   ├── evaluate_prompts.py    # Benchmark runner for V1-V4
│   ├── personas.py            # Scripted candidate personas
│   ├── prompt_test_cases.csv  # 10 standardized test cases
│   ├── prompt_evaluation_report.md # Generated evaluation report
│   ├── prompt_evaluation_results.json # Raw benchmark telemetry
│   └── run_eval.py            # End-to-end simulated session runner
├── frontend/                  # Modern AI SaaS Web Interface
│   ├── index.html             # Application views (Setup, Interview, Report)
│   ├── script.js              # State controller, Speech API & dynamic rendering
│   └── style.css              # Clean, responsive SaaS dashboard styling
├── tests/                     # 179 Automated Unit Tests
│   ├── conftest.py            # Test fixtures & FakeLLM client
│   ├── test_interviewace_scoring.py # Rubric validation & analytics tests
│   ├── test_feedback.py       # Feedback generation & caching tests
│   ├── test_graph.py          # LangGraph state machine tests
│   ├── test_limits.py         # Rate limiting & upload ceiling tests
│   ├── test_llm_client.py     # Key failover & error handling tests
│   ├── test_matching.py       # BM25 algorithm tests
│   ├── test_persona.py        # Candidate persona detection tests
│   ├── test_resume_checks.py  # Resume sanity check tests
│   ├── test_resume_score.py   # Deterministic resume scoring tests
│   ├── test_role_detection.py # Role inference tests
│   └── test_session_lifecycle.py # Interview session lifecycle tests
├── .env.example               # Template for environment secrets
├── pyproject.toml             # Project dependencies & tool configurations
└── README.md                  # Comprehensive documentation
```

---

## 🔒 Security, Privacy & Integrity

- **Zero Client-Side Secrets:** API keys are never exposed to the frontend; all LLM invocations occur inside server-side FastAPI endpoints.
- **No Long-Term User PII:** Checkpointed interview sessions live in temporary local storage only for the duration of the interview; no user account or login is required.
- **Anti-Injection Protection:** Conversation transcripts are serialized using native JSON structures rather than naive delimiter interpolation, preventing candidates from injecting forged turn boundaries.
- **Strict Server-Side Bounds:** All scores are clamped $[0, 5]$ and totals are recomputed by Python code, neutralizing prompt manipulation attacks.

---

## ⚠️ Limitations & Future Work

1. **Local Speech API Browser Dependency:** The Web Speech API is supported natively on Google Chrome and Microsoft Edge, but not in Mozilla Firefox. A future version can integrate server-side Whisper speech-to-text.
2. **Video & Body Language Analysis:** Future iterations could incorporate computer vision for eye contact, posture, and pacing analysis.
3. **Company-Specific Placements:** Future expansion can support customized question banks tailored to specific university recruitment drives (e.g. FAANG, FinTech, Big 4 Consultancies).

---

## 🎓 Academic Submission Details

- **Student Project:** InterviewAce — Problem C10 (Mock Interview Question and Feedback Tool)
- **Course:** Prompt Engineering & LLM Application Development
- **Status:** Milestone 1 Complete (100% requirements verified)
