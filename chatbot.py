from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import openai
import os
import logging
import requests

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

app = FastAPI()
router = APIRouter()

# ---------------------------
# Schemas
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
    description: Optional[str] = None

class SessionSummary(BaseModel):
    topic: str
    weak_areas: List[str]
    target_level: str

class ChatbotInput(BaseModel):
    user_input: str
    current_phase: Phase
    session_summary: SessionSummary
    message_history: Optional[List[Dict[str, str]]] = []

class ActionSuggestion(BaseModel):
    type: str
    reason: Optional[str] = None

class ChatbotResponse(BaseModel):
    message: str
    follow_up_question: Optional[str] = None
    action_suggestion: Optional[ActionSuggestion] = None
    context_update_required: bool = False
    learning_concepts_covered: Optional[List[str]] = None
    new_user_goal: Optional[str] = None

# ---------------------------
# Helpers
# ---------------------------
def get_context_slice() -> Dict[str, Any]:
    try:
        response = requests.get(f"{CONTEXT_BASE_URL}/api/context/slice")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Context fetch failed: {e}")
        return {}

def build_prompt(data: ChatbotInput, context: Dict[str, Any]) -> str:
    ctx = []
    if context.get("current_topic"):
        ctx.append(f"Current topic: {context['current_topic']}")
    if context.get("user_goals"):
        ctx.append(f"User goals: {', '.join(context['user_goals'])}")
    if context.get("weak_areas"):
        ctx.append(f"Weak areas: {', '.join(context['weak_areas'])}")
    if context.get("emphasized_facts"):
        ctx.append(f"Emphasized facts: {', '.join(context['emphasized_facts'])}")
    if context.get("preferred_learning_styles"):
        ctx.append(f"Preferred learning styles: {', '.join(context['preferred_learning_styles'])}")

    recent_messages = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in data.message_history[-3:]
    ])

    ctx_text = "\n".join(ctx)

    base = f"""
You are Arlo, an expert AI tutor.
Avoid greetings or filler phrases like 'Hey there' or 'Nice to see you.'
Respond directly and clearly like a human teacher sitting next to the student.
Focus first on the student’s current question, and supplement only with relevant context.
{ctx_text}

Recent conversation:
{recent_messages}
"""

    user_input = data.user_input.strip()
    phase = data.current_phase.phase
    payload = data.current_phase.payload or PhasePayload()

    if phase == "flashcards" and payload.question:
        return base + f"""
Flashcard:
Q: {payload.question}
User answered: {payload.user_answer}
Follow-up input: {user_input}
Give concise correction or reinforcement.
"""

    elif phase == "feynman":
        return base + f"""
The student is explaining aloud.
They said: "{user_input}"
Help them identify gaps and improve the explanation.
"""

    elif phase == "quiz" and payload.question:
        return base + f"""
Quiz:
Q: {payload.question}
User answered: {payload.user_answer}
Follow-up input: {user_input}
Correct and explain.
"""

    elif phase == "blurting":
        return base + f"""
The student is blurting — trying to recall everything about the topic.
They said: "{user_input}"
Point out missing info and help reinforce.
"""

    else:
        return base + f"""
Student said: "{user_input}"
Respond with helpful explanation or next step.
"""

def call_gpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error calling GPT: {e}")
        return "Sorry, I had trouble generating a response."

@router.post("/api/chatbot", response_model=ChatbotResponse)
def chatbot_handler(data: ChatbotInput):
    logger.info("Chatbot request received")
    context = get_context_slice()
    prompt = build_prompt(data, context)
    logger.debug(f"Prompt:\n{prompt}")
    gpt_reply = call_gpt(prompt)

    action = None
    if data.current_phase.phase in ["flashcards", "quiz"] and "correct" in gpt_reply.lower():
        action = ActionSuggestion(type="next_phase", reason="Answer was correct")

    return ChatbotResponse(
        message=gpt_reply,
        follow_up_question=None,
        action_suggestion=action,
        context_update_required=False
    )

app.include_router(router)
