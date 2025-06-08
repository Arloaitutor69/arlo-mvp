from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict
import openai
import os
import logging

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

# FastAPI app and router setup
app = FastAPI()
router = APIRouter()

# ---------------------------
# Pydantic Schemas
# ---------------------------
class PhasePayload(BaseModel):
    question: Optional[str] = None
    user_answer: Optional[str] = None
    subtopic: Optional[str] = None

class Phase(BaseModel):
    phase: str
    tool: str
    lovable_component: str
    payload: Optional[PhasePayload] = None

class SessionSummary(BaseModel):
    topic: str
    weak_areas: List[str]
    target_level: str

class ChatbotInput(BaseModel):
    user_input: str
    current_phase: Phase
    session_summary: SessionSummary

class ActionSuggestion(BaseModel):
    type: str  # e.g., "next_phase"
    reason: Optional[str] = None

class ChatbotResponse(BaseModel):
    message: str
    follow_up_question: Optional[str] = None
    action_suggestion: Optional[ActionSuggestion] = None

# ---------------------------
# Prompt Builder
# ---------------------------
def build_prompt(data: ChatbotInput) -> str:
    base = f"""
You are Arlo, a friendly and brilliant AI tutor helping a high school student learn {data.session_summary.topic}.
They are currently in the {data.current_phase.phase} phase.
Use a warm, human-like tone. Respond like you're sitting next to them.
"""

    user_input = data.user_input.strip()
    phase = data.current_phase.phase
    payload = data.current_phase.payload or PhasePayload()

    if phase == "flashcards" and payload.question:
        return base + f'''
Student just answered a flashcard:
Q: {payload.question}
A: {payload.user_answer}

Give helpful, encouraging feedback.
Then respond to their input: "{user_input}"'''

    elif phase == "feynman":
        return base + f'''
The student is trying to explain the topic out loud using the Feynman technique.
They said: "{user_input}"

Evaluate their explanation. Be supportive. Ask them to clarify any weak points.'''

    elif phase == "teaching":
        return base + f'''
The student is currently learning. They said: "{user_input}"

Respond with an explanation or follow-up question.'''

    elif phase == "quiz" and payload.question:
        return base + f'''
The student is answering a quiz:
Q: {payload.question}
A: {payload.user_answer}

Give correction and encouragement.
Respond to their input: "{user_input}"'''

    elif phase == "blurting":
        return base + f'''
The student is using the blurting technique, trying to recall everything about the topic.
They said: "{user_input}"

Identify missing pieces and encourage memory reinforcement.'''

    else:
        return base + f'''
Student said: "{user_input}"
Respond appropriately.'''

# ---------------------------
# GPT API Call
# ---------------------------
def call_gpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling GPT: {e}")
        return "Sorry, there was a problem generating a response. Please try again."

# ---------------------------
# Endpoint
# ---------------------------
@router.post("/api/chatbot", response_model=ChatbotResponse)
def chatbot_handler(data: ChatbotInput):
    logger.info("Received request for chatbot")
    prompt = build_prompt(data)
    logger.debug(f"Prompt sent to GPT:\n{prompt}")

    gpt_reply = call_gpt(prompt)
    logger.info("Received response from GPT")

    # Heuristic suggestion to move on if feedback contains "correct"
    action = None
    if data.current_phase.phase in ["flashcards", "quiz"] and "correct" in gpt_reply.lower():
        action = ActionSuggestion(
            type="next_phase",
            reason="Student seems to understand this concept."
        )

    return ChatbotResponse(
        message=gpt_reply,
        follow_up_question=None,
        action_suggestion=action
    )

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

# ---------------------------
# For Local Debugging
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
