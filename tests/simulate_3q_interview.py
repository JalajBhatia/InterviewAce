import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("--- Step 1: Start 3-Question Session ---")
start_resp = client.post("/start_session", json={"role": "Backend Developer", "question_count": 3})
assert start_resp.status_code == 200, f"start error: {start_resp.text}"
start_data = start_resp.json()
session_id = start_data["session_id"]
print("Session ID:", session_id)
print("Configured max questions:", start_data["max_questions"])
print("Opening message:", start_data["initial_message"][:100], "...")

print("\n--- Step 2: Answer Question 1 ---")
q1_resp = client.post("/chat", json={
    "session_id": session_id,
    "user_message": "To design the registration endpoint, I would expose POST /api/v1/auth/register taking email and password. I would validate the input, hash the password using Argon2 or bcrypt with a salt, store the user in PostgreSQL with a unique constraint on email, and return a JWT token with HTTP-only cookie."
})
assert q1_resp.status_code == 200
q1_data = q1_resp.json()
print("Q1 questions_asked:", q1_data["questions_asked"], "action:", q1_data["action"], "is_over:", q1_data["is_interview_over"])
print("Q1 Score:", q1_data["evaluation"]["rubric"]["total_score"] if q1_data["evaluation"] else "None")
print("Interviewer next question:", q1_data["agent_message"][:120], "...")

print("\n--- Step 3: Answer Question 2 ---")
q2_resp = client.post("/chat", json={
    "session_id": session_id,
    "user_message": "For connection pooling, I configure min and max pool sizes based on database server limits and traffic concurrency. For query optimization, I inspect execution plans using EXPLAIN ANALYZE, ensure indexes on foreign keys and filter columns, and avoid N+1 queries by using eager loading."
})
assert q2_resp.status_code == 200
q2_data = q2_resp.json()
print("Q2 questions_asked:", q2_data["questions_asked"], "action:", q2_data["action"], "is_over:", q2_data["is_interview_over"])
print("Q2 Score:", q2_data["evaluation"]["rubric"]["total_score"] if q2_data["evaluation"] else "None")
print("Interviewer next question:", q2_data["agent_message"][:120], "...")

print("\n--- Step 4: Answer Question 3 ---")
q3_resp = client.post("/chat", json={
    "session_id": session_id,
    "user_message": "For microservices architecture and inter-service communication, I use gRPC with Protocol Buffers for internal low-latency synchronous calls, and Apache Kafka or RabbitMQ for asynchronous event-driven workflows with circuit breakers and dead-letter queues."
})
assert q3_resp.status_code == 200
q3_data = q3_resp.json()
print("Q3 questions_asked:", q3_data["questions_asked"], "action:", q3_data["action"], "is_over:", q3_data["is_interview_over"])
print("Q3 Score:", q3_data["evaluation"]["rubric"]["total_score"] if q3_data["evaluation"] else "None")
print("Agent prompt (Closing Question):", q3_data["agent_message"])

print("\n--- Step 5: Answer Closing Question ---")
close_resp = client.post("/chat", json={
    "session_id": session_id,
    "user_message": "No thank you, that covers all my questions. Thank you for your time!"
})
assert close_resp.status_code == 200
close_data = close_resp.json()
print("Closing response flags:")
print("  is_interview_over:", close_data["is_interview_over"])
print("  is_completed:", close_data["is_completed"])
print("  action:", close_data["action"])
print("  evaluation:", close_data["evaluation"])
print("  questions_asked:", close_data["questions_asked"])
print("  agent_message:", close_data["agent_message"])

assert close_data["is_interview_over"] is True
assert close_data["is_completed"] is True
assert close_data["action"] == "close"
assert close_data["evaluation"] is None
assert close_data["questions_asked"] <= 3

print("\n--- Step 6: Generate and Fetch Final Report (/feedback) ---")
fb_resp = client.post("/feedback", json={"session_id": session_id})
assert fb_resp.status_code == 200
fb_data = fb_resp.json()
summary = fb_data["feedback"]["performance_summary"]
print("Final Report Performance Summary:")
print("  Questions answered:", summary["questions_answered"])
print("  Max possible marks:", summary["max_possible_marks"])
print("  Total marks obtained:", summary["total_marks_obtained"])
print("  Overall percentage:", summary["overall_percentage"], "%")
print("  Average score:", summary["average_score"], "/ 20")
print("  Readiness verdict:", fb_data["feedback"]["verdict"])

assert summary["questions_answered"] == 3
assert summary["max_possible_marks"] == 60

print("\n--- Step 7: Test Duplicate Feedback Request ---")
fb_resp2 = client.post("/feedback", json={"session_id": session_id})
assert fb_resp2.status_code == 200
summary2 = fb_resp2.json()["feedback"]["performance_summary"]
assert summary2["questions_answered"] == summary["questions_answered"]
assert summary2["total_marks_obtained"] == summary["total_marks_obtained"]
print("Duplicate feedback request successfully returned identical cached data without corruption.")

print("\n=======================================================")
print("ALL VERIFICATIONS PASSED: 3-QUESTION FLOW 100% HEALTHY!")
print("=======================================================")
