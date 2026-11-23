"""
InterviewAce Rubric Scoring and Performance Analytics Engine

Implements the fixed 4-criteria evaluation rubric:
1. Correctness: 0–5
2. Relevance: 0–5
3. Clarity: 0–5
4. Completeness: 0–5

Total Score: 0–20
Percentage: (total_score / 20) * 100

Ensures backend validates all scores, enforces bounds, and calculates totals
rather than blindly trusting LLM output.
"""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import AnswerEvaluation, PerformanceSummary, RubricScores


def clamp_score(val: Any) -> int:
    """Safely clamp any input value to an integer between 0 and 5."""
    try:
        score = int(round(float(val)))
        return max(0, min(5, score))
    except (ValueError, TypeError):
        return 0


def validate_and_compute_rubric(rubric_data: RubricScores | Mapping[str, Any] | None) -> RubricScores:
    """
    Validate, clamp, and compute total and percentage for the rubric.
    Never blindly trusts an AI-generated total.
    """
    if rubric_data is None:
        return RubricScores(
            correctness=0,
            relevance=0,
            clarity=0,
            completeness=0,
            total_score=0,
            percentage=0.0,
        )

    if isinstance(rubric_data, RubricScores):
        c = clamp_score(rubric_data.correctness)
        r = clamp_score(rubric_data.relevance)
        cl = clamp_score(rubric_data.clarity)
        comp = clamp_score(rubric_data.completeness)
    elif isinstance(rubric_data, dict):
        c = clamp_score(rubric_data.get("correctness", 0))
        r = clamp_score(rubric_data.get("relevance", 0))
        cl = clamp_score(rubric_data.get("clarity", 0))
        comp = clamp_score(rubric_data.get("completeness", 0))
    else:
        c = clamp_score(getattr(rubric_data, "correctness", 0))
        r = clamp_score(getattr(rubric_data, "relevance", 0))
        cl = clamp_score(getattr(rubric_data, "clarity", 0))
        comp = clamp_score(getattr(rubric_data, "completeness", 0))

    total = c + r + cl + comp
    percentage = round((total / 20.0) * 100.0, 1)

    return RubricScores(
        correctness=c,
        relevance=r,
        clarity=cl,
        completeness=comp,
        total_score=total,
        percentage=percentage,
    )


def sanitize_answer_evaluation(eval_data: AnswerEvaluation | Mapping[str, Any] | None) -> AnswerEvaluation | None:
    """Sanitize and validate an AnswerEvaluation object."""
    if eval_data is None:
        return None

    raw_q_num = None
    raw_paste = False
    if isinstance(eval_data, AnswerEvaluation):
        validated_rubric = validate_and_compute_rubric(eval_data.rubric)
        strengths = [s.strip() for s in eval_data.strengths if str(s).strip()] or ["Good effort"]
        improvements = [i.strip() for i in eval_data.areas_for_improvement if str(i).strip()] or ["Add more technical depth"]
        improved = (eval_data.suggested_improved_answer or "").strip()
        follow_up = (eval_data.follow_up_question or "").strip()
        raw_q_num = getattr(eval_data, "question_number", None)
        raw_paste = getattr(eval_data, "paste_detected", False)
    elif isinstance(eval_data, dict):
        validated_rubric = validate_and_compute_rubric(eval_data.get("rubric"))
        strengths = [s.strip() for s in eval_data.get("strengths", []) if str(s).strip()] or ["Good effort"]
        improvements = [i.strip() for i in eval_data.get("areas_for_improvement", []) if str(i).strip()] or ["Add more technical depth"]
        improved = str(eval_data.get("suggested_improved_answer", "")).strip()
        follow_up = str(eval_data.get("follow_up_question", "")).strip()
        raw_q_num = eval_data.get("question_number")
        raw_paste = eval_data.get("paste_detected", False)
    else:
        return None

    parsed_q_num = None
    if raw_q_num is not None:
        try:
            val = int(raw_q_num)
            if val >= 0:
                parsed_q_num = val
        except (ValueError, TypeError):
            parsed_q_num = None

    return AnswerEvaluation(
        rubric=validated_rubric,
        strengths=strengths,
        areas_for_improvement=improvements,
        suggested_improved_answer=improved,
        follow_up_question=follow_up,
        question_number=parsed_q_num,
        paste_detected=bool(raw_paste),
    )


def compute_performance_summary(
    evaluations: list[dict | AnswerEvaluation],
    overall_strengths: list[str] | None = None,
    areas_that_need_improvement: list[str] | None = None,
    personalized_recommendations: list[str] | None = None,
    max_questions: int | None = None,
) -> PerformanceSummary:
    """
    Aggregate all answer evaluations across an interview session into a
    comprehensive performance report with actual marks, criteria analytics,
    and placement guidance.

    Guarantees:
    - Each distinct question contributes at most 20 marks.
    - Follow-up turns on the same question update that question's score rather than
      adding multiple 20-mark chunks that exceed max_questions * 20 (e.g. 142/100).
    - Maximum possible marks is strictly question_count * 20.
    - Missing, invalid, or out-of-order question numbers are resolved gracefully.
    """
    valid_evals = []
    for item in evaluations:
        sanitized = sanitize_answer_evaluation(item)
        if sanitized is not None:
            valid_evals.append(sanitized)

    if not valid_evals:
        return PerformanceSummary(
            questions_answered=0,
            total_marks_obtained=0,
            max_possible_marks=0,
            overall_percentage=0.0,
            average_score=0.0,
            strongest_criterion="N/A",
            weakest_criterion="N/A",
            criterion_totals={"correctness": 0, "relevance": 0, "clarity": 0, "completeness": 0},
            criterion_averages={"correctness": 0.0, "relevance": 0.0, "clarity": 0.0, "completeness": 0.0},
            overall_strengths=overall_strengths or ["Participated in the interview."],
            areas_that_need_improvement=areas_that_need_improvement or ["Answer more questions for detailed feedback."],
            personalized_recommendations=personalized_recommendations or ["Practice answering core role questions."],
        )

    # Deduplicate / resolve by question_number so follow-up answers on the same question
    # update the question's score rather than inflating the total beyond question_count * 20.
    by_question: dict[int, AnswerEvaluation] = {}
    auto_slot = 1
    for e in valid_evals:
        q_num = getattr(e, "question_number", None)
        if q_num is not None and isinstance(q_num, int):
            if q_num <= 0:
                continue  # Warm-up introduction answers do not occupy technical question slots
            target_slot = min(q_num, max_questions) if max_questions else q_num
        else:
            target_slot = min(auto_slot, max_questions) if max_questions else auto_slot

        if target_slot in by_question:
            existing = by_question[target_slot]
            is_pasted = getattr(e, "paste_detected", False)
            # Pasted answers must remain 0 and not reset or replace an earned score
            if not is_pasted:
                if e.rubric.total_score >= existing.rubric.total_score:
                    by_question[target_slot] = e
        else:
            by_question[target_slot] = e

        auto_slot = max(auto_slot, target_slot + 1)

    effective_evals = [by_question[k] for k in sorted(by_question.keys())]
    if max_questions is not None and len(effective_evals) > max_questions:
        effective_evals = effective_evals[:max_questions]

    count = len(effective_evals)
    correctness_tot = sum(e.rubric.correctness for e in effective_evals)
    relevance_tot = sum(e.rubric.relevance for e in effective_evals)
    clarity_tot = sum(e.rubric.clarity for e in effective_evals)
    completeness_tot = sum(e.rubric.completeness for e in effective_evals)

    total_marks = correctness_tot + relevance_tot + clarity_tot + completeness_tot
    # Each evaluated answer contributes at most 20 marks.
    # Maximum possible marks for the evaluated questions is count * 20.
    max_possible = count * 20
    overall_pct = round((total_marks / max_possible) * 100.0, 1) if max_possible > 0 else 0.0
    avg_score = round(total_marks / count, 1) if count > 0 else 0.0

    criterion_totals = {
        "correctness": correctness_tot,
        "relevance": relevance_tot,
        "clarity": clarity_tot,
        "completeness": completeness_tot,
    }

    criterion_averages = {
        "correctness": round(correctness_tot / count, 1) if count > 0 else 0.0,
        "relevance": round(relevance_tot / count, 1) if count > 0 else 0.0,
        "clarity": round(clarity_tot / count, 1) if count > 0 else 0.0,
        "completeness": round(completeness_tot / count, 1) if count > 0 else 0.0,
    }

    criterion_display = {
        "correctness": "Correctness & Conceptual Soundness",
        "relevance": "Relevance & Direct Answers",
        "clarity": "Clarity & Logical Structure",
        "completeness": "Completeness & Depth",
    }

    # Identify strongest and weakest criterion
    sorted_criteria = sorted(criterion_totals.items(), key=lambda kv: kv[1])
    weakest_key = sorted_criteria[0][0]
    strongest_key = sorted_criteria[-1][0]

    # Collect strengths and improvements from evals if not provided
    if not overall_strengths:
        all_s = []
        for e in valid_evals:
            all_s.extend(e.strengths)
        # Deduplicate while preserving order
        seen = set()
        dedup_s = []
        for s in all_s:
            if s.lower() not in seen:
                seen.add(s.lower())
                dedup_s.append(s)
        overall_strengths = dedup_s[:4] or ["Demonstrated willingness to tackle technical questions."]

    if not areas_that_need_improvement:
        all_i = []
        for e in valid_evals:
            all_i.extend(e.areas_for_improvement)
        seen = set()
        dedup_i = []
        for item in all_i:
            if item.lower() not in seen:
                seen.add(item.lower())
                dedup_i.append(item)
        areas_that_need_improvement = dedup_i[:4] or ["Provide more concrete examples and address edge cases."]

    if not personalized_recommendations:
        personalized_recommendations = [
            f"Focus on strengthening {criterion_display[weakest_key]} through targeted practice problems.",
            "Use structured answering frameworks: STAR (Situation, Task, Action, Result) for behavioral questions, and Input-Approach-Complexity for technical coding.",
            "Review trade-offs and edge cases for core data structures and system components before campus placement drives.",
        ]

    return PerformanceSummary(
        questions_answered=count,
        total_marks_obtained=total_marks,
        max_possible_marks=max_possible,
        overall_percentage=overall_pct,
        average_score=avg_score,
        strongest_criterion=criterion_display[strongest_key],
        weakest_criterion=criterion_display[weakest_key],
        criterion_totals=criterion_totals,
        criterion_averages=criterion_averages,
        overall_strengths=overall_strengths,
        areas_that_need_improvement=areas_that_need_improvement,
        personalized_recommendations=personalized_recommendations,
    )
