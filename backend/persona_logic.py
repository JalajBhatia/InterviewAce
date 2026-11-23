"""
Persona detection and handling logic for adaptive interview behavior.
"""

import re

# Phrases that mean "I want to stop now", matched against normalised text.
END_INTENT_PHRASES = (
    "end interview", "end the interview", "end this interview",
    "stop interview", "stop the interview", "stop this interview",
    "finish interview", "finish the interview",
    "end session", "end the session",
    "quit interview", "exit interview",
    "end it here", "stop here",
    "wrap up", "wrap this up",
    "im done", "i am done", "were done", "we are done",
    "thats all", "that is all", "thats it", "that is it",
    "no more questions",
)

CLOSING_DECLINE_PHRASES = (
    "no", "nope", "nah", "none",
    "no thank you", "no thanks", "no thank u",
    "no questions", "no question", "no more questions",
    "i have no questions", "i don't have any questions", "i dont have any questions",
    "i do not have any questions", "no questions from me",
    "no questions from my side", "no questions from my end",
    "nothing from my side", "nothing from me", "nothing else",
    "all good", "im good", "i'm good", "i am good",
    "all set", "all clear", "that covers it", "that covers everything",
    "that covers all my questions", "that covers all of my questions", "that covers it all",
    "thats all", "that's all", "that is all",
    "no thats all", "no that's all", "no that is all",
    "thank you", "thanks", "thank you so much", "thanks so much",
    "no thank you so much", "no thanks so much",
    "thank you for your time", "thanks for your time",
    "thank you for the opportunity", "thanks for the opportunity",
    "no further questions", "i have no further questions",
)

# An end request is a short command. Anything longer is almost certainly the
# candidate telling a story ("at my last job I had to end the interview early
# because the manager..."), which must not terminate the session.
MAX_END_REQUEST_WORDS = 8


def is_closing_decline(user_message: str) -> bool:
    """True if candidate response in closing stage indicates having no further questions."""
    if not user_message:
        return True
    cleaned = user_message.lower().replace("'", "").replace("’", "")
    normalized = " ".join(re.sub(r"[^a-z0-9\s]", " ", cleaned).split())
    if not normalized:
        return True
    words = normalized.split()
    if len(words) <= 1 and words[0] in ("no", "nope", "nah", "none", "thanks", "done"):
        return True
    if len(words) <= 25:
        return any(phrase.replace("'", "") in normalized for phrase in CLOSING_DECLINE_PHRASES)
    return False


def is_end_request(user_message: str) -> bool:
    """
    True if the candidate is asking to stop the interview.

    Rule-based on purpose, matching how personas are detected: an extra LLM call
    per turn would add latency to a voice app and give a non-deterministic answer
    to a question that is cheap to decide directly.

    Two guards keep false positives down - a word limit, and rejecting questions,
    since "should I end the interview?" is asking, not instructing.
    """
    if not user_message or "?" in user_message:
        return False

    # Drop apostrophes first so "that's all" -> "thats all" rather than
    # "that s all", then reduce everything else to single-spaced words.
    normalized = user_message.lower().replace("'", "").replace("’", "")
    normalized = " ".join(re.sub(r"[^a-z0-9\s]", " ", normalized).split())

    if not normalized or len(normalized.split()) > MAX_END_REQUEST_WORDS:
        return False

    return any(phrase in normalized for phrase in END_INTENT_PHRASES)


def detect_persona(user_message: str, conversation_history: list) -> str:
    """
    Detects user persona based on their response patterns.
    
    Returns: "confused", "efficient", "chatty", "edge", or "normal"
    """
    message_lower = user_message.lower().strip()
    word_count = len(user_message.split())
    
    # Edge Case Detection
    edge_keywords = ["make me ceo", "give me job", "hire me now", "your weights", "your model", 
                     "banana", "asdfgh", "blah blah", "random gibberish"]
    if any(keyword in message_lower for keyword in edge_keywords):
        return "edge"
    
    if word_count < 3 and not any(word in message_lower for word in ["yes", "no", "okay", "sure"]):
        return "edge"
    
    # Confused Detection
    confused_keywords = ["i don't know", "not sure", "what do you mean", "don't understand", 
                        "confused", "unclear", "help", "what is", "explain"]
    if any(keyword in message_lower for keyword in confused_keywords):
        return "confused"
    
    # Very short answers repeatedly
    if word_count <= 5:
        # Check history for pattern of short answers
        recent_user_messages = [msg["content"] for msg in conversation_history[-6:] if msg["role"] == "user"]
        if len(recent_user_messages) >= 2:
            avg_length = sum(len(m.split()) for m in recent_user_messages) / len(recent_user_messages)
            if avg_length < 8:
                return "confused"
    
    # Efficient Detection
    if 5 < word_count <= 25:
        # Concise but complete
        return "efficient"
    
    # Chatty Detection
    if word_count > 100:
        return "chatty"
    
    # Check for off-topic rambling
    if word_count > 60:
        offtopic_keywords = ["by the way", "also", "speaking of", "reminds me", "funny story"]
        if any(keyword in message_lower for keyword in offtopic_keywords):
            return "chatty"
    
    return "normal"


def get_persona_instruction(persona: str) -> str:
    """
    Returns specific instructions for handling detected persona.
    """
    instructions = {
        "confused": """
The user seems confused or uncertain. 
- Simplify your next question
- Provide a bit more context
- Be supportive and encouraging
- Consider rephrasing if they're stuck on the same question
""",
        "efficient": """
The user is being concise and efficient.
- Keep your questions brief and to the point
- Don't over-explain
- Move through topics quickly
- Respect their time
""",
        "chatty": """
The user is giving very long or off-topic answers.
- Politely acknowledge their answer
- Summarize the key point you heard
- Redirect to a specific, focused question
- Set clear boundaries: "Let's focus on..."
""",
        "edge": """
The user gave an irrelevant, nonsense, or inappropriate response.
- Politely set a boundary
- Explain what you can and cannot do
- Redirect back to the interview
- Example: "I can't do that, but I can help you practice for this role. Let's continue..."
""",
        "normal": """
The user is responding normally.
- Continue with standard interview flow
- Ask follow-ups if answer lacks depth
- Move to next topic if answer is complete
"""
    }
    return instructions.get(persona, instructions["normal"])
