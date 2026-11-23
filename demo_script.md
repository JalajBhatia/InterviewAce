# Demo Script: InterviewAce

**Academic Mini-Project:** Problem C10 — Mock Interview Question and Feedback Tool  
**Duration:** 5–7 Minutes  

---

## 1. Introduction & Mission (45 seconds)

> "Hello everyone. Today I'm demonstrating **InterviewAce**, an AI-powered mock interview and personalized feedback assistant built specifically for university students preparing for campus placements and technical internships.
>
> Many students struggle in placements not because they lack coding skills, but because they lack structured communication, fail to cover edge cases, or freeze during follow-up questions.
>
> InterviewAce solves this by providing:
> 1. Role-specific technical and HR mock interviews.
> 2. A fixed 4-criteria evaluation rubric scored out of 20 marks on every single answer.
> 3. Real-time feedback with constructive suggestions and model exemplar answers.
> 4. An adaptive state machine that asks intelligent follow-up questions based on the candidate's answers.
> 5. A comprehensive final performance report with criteria analytics."

---

## 2. Architecture & Prompt Engineering Highlights (1 minute)

> "From a prompt engineering perspective, this project demonstrates:
> - **Prompt Versioning & Evolution:** We designed 4 distinct prompt iterations — from V1 Baseline, to V2 Structured, V3 Improved with strict anchor criteria, up to V4 Final Production.
> - **Mitigating Verbosity & Hallucination Bias:** Standard LLMs tend to award high scores to long answers regardless of correctness. Our prompts enforce strict scoring anchors and penalize ungrounded or fabricated claims.
> - **Server-Side Integrity:** The FastAPI backend never blindly trusts LLM arithmetic; it validates all criteria within 0 to 5 and calculates totals and percentages mathematically.
> - **Multi-Turn State Machine:** Built on LangGraph, so the agent counts actual questions asked and adaptively decides whether to `advance`, `follow_up`, `rephrase`, or `redirect`."

---

## 3. Live Walkthrough & Demonstration Flow (4 minutes)

### Step 1: Session Setup Screen
1. Open `http://localhost:8000/app/` in Google Chrome or Microsoft Edge.
2. Highlight the clean AI SaaS dashboard layout.
3. In the setup panel:
   - Select **Target Role:** `Software Development Intern`
   - Select **Track:** `Technical Interview`
   - Select **Difficulty:** `Medium (Standard Placement Level)`
   - Select **Question Count:** `3 Questions`
   - Select **Prompt Version:** `V4 — Final Production Prompt (Recommended)`
4. Show the **Fixed 4-Criteria Rubric** preview card (Correctness, Relevance, Clarity, Completeness).
5. Click **"Start Interview Session 🚀"**.

### Step 2: Answering with Real-Time Rubric Evaluation
1. **First Question from AI:**
   - The AI greets the candidate professionally and asks an opening technical question (e.g. difference between list and tuple, or how hash maps handle collisions).
2. **Submit a Good Answer:**
   - Type or speak a concise, accurate answer explaining mutability and hashability.
   - Click **"Submit Answer for Evaluation"**.
   - **Observe:**
     * The AI generates a **Rubric Evaluation Card** directly under your answer.
     * Shows: Correctness (5/5), Relevance (5/5), Clarity (5/5), Completeness (4/5), Turn Score: 19/20 (95%).
     * Shows positive **Strengths** citing your words.
     * Shows **Areas for Improvement**.
     * Shows a **Suggested Exemplar Answer** illustrating model placement communication.
     * Topbar live score tracker updates immediately.
3. **Adaptive Follow-Up Question:**
   - The AI asks a follow-up probing an edge case or time complexity before advancing to the next question.
4. **Submit a Weak / Partial Answer:**
   - On the next turn, type a vague or incomplete answer.
   - **Observe:**
     * The rubric card reflects lower scores (e.g., Completeness 2/5, Correctness 3/5).
     * The AI provides clear corrective guidance and an improved answer.

### Step 3: Final Performance Report
1. Click **"Finish Interview & Get Report 📊"** (or complete all configured questions).
2. The UI transitions to the **Comprehensive Placement Performance Summary**:
   - **Executive Metric Cards:** Questions Attempted, Total Marks Obtained, Overall Percentage, and Campus Placement Readiness Badge (e.g., `Placement Ready ⭐` or `Promising Candidate 👍`).
   - **Criterion Analytics:** Visual progress bars for Correctness, Relevance, Clarity, and Completeness.
   - **Strongest vs Weakest Criterion:** Highlighting the student's top competency and highest-leverage area for placement revision.
   - **Hiring Verdict & Justification:** Clear recommendation (`Hire`, `Strong Hire`, `Maybe`, `No Hire`).
   - **Personalized Recommendations:** Concrete study and communication next steps.

---

## 4. Benchmark & Testing Demonstration (1 minute)

> "To verify our prompt engineering rigor:
> - We maintain a standardized benchmark of 10 fictional candidate test cases in `evals/prompt_test_cases.csv`.
> - Running `python -m evals.evaluate_prompts` demonstrates that V4 achieves 100% expected range accuracy and a +16/20 discrimination index between exemplary and irrelevant answers.
> - The repository includes 179 automated unit tests verifying score clamping, total recalculation, report aggregation, and state persistence."

---

## 5. Conclusion (30 seconds)

> "InterviewAce transforms mock interview preparation from an informal chat into an objective, standardized, and educational placement training tool. Thank you!"
