from fastapi import APIRouter, FastAPI, HTTPException, Request
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
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

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
    source: Optional[str] = None
    user_id: Optional[str] = None

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
def extract_user_id(request: Request, data: ChatbotInput) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    elif data.source and str(data.source).startswith("user:"):
        return data.source.replace("user:", "")
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

def get_context_slice(user_id: str) -> Dict[str, Any]:
    try:
        response = requests.get(f"{CONTEXT_API}/api/context/slice?user_id={user_id}")
        response.raise_for_status()
        raw = response.json()
        return {
            "current_topic": raw.get("current_topic"),
            "user_goals": raw.get("user_goals", [])[:2],
            "weak_areas": raw.get("weak_areas", [])[:2],
            "emphasized_facts": raw.get("emphasized_facts", [])[:2],
            "preferred_learning_styles": raw.get("preferred_learning_styles", [])[:1]
        }
    except Exception as e:
        logger.warning(f"Context fetch failed: {e}")
        return {}

def build_prompt(data: ChatbotInput, context: Dict[str, Any]) -> str:
    ctx = []
    if context.get("current_topic"):
        ctx.append(f"Current topic: {context['current_topic']}")
    if context.get("user_goals"):
        ctx.append(f"Goals: {', '.join(context['user_goals'])}")
    if context.get("weak_areas"):
        ctx.append(f"Weak areas: {', '.join(context['weak_areas'])}")
    if context.get("emphasized_facts"):
        ctx.append(f"Focus points: {', '.join(context['emphasized_facts'])}")
    if context.get("preferred_learning_styles"):
        ctx.append(f"Style: {context['preferred_learning_styles'][0]}")

    recent_messages = "\n".join([
        f"{msg['role']}: {msg['content']}" for msg in data.message_history[-3:]
    ])

    ctx_text = "\n".join(ctx)
    base = f"""
You are Arlo, an expert AI tutor.
Skip greetings and filler. Focus on instruction.
{ctx_text}

Recent conversation:
{recent_messages}
"""

    user_input = data.user_input.strip()
    phase = data.current_phase.phase
    payload = data.current_phase.payload or PhasePayload()
    description = data.current_phase.description or ""

    if phase == "flashcards" and payload.question:
        return base + f"""
Flashcard:
Q: {payload.question}
User answered: {payload.user_answer}
Follow-up input: {user_input}
Correct or reinforce briefly.
"""
    elif phase == "feynman":
        return base + f"""
The student is explaining aloud.
They said: \"{user_input}\"
Help identify gaps and improve understanding.
"""
    elif phase == "quiz" and payload.question:
        return base + f"""
Quiz:
Q: {payload.question}
User answered: {payload.user_answer}
Follow-up input: {user_input}
Correct and explain clearly.
"""
    elif phase == "blurting":
        return base + f"""
The student is blurting — recalling everything.
They said: \"{user_input}\"
Point out what’s missing and reinforce.
"""
    elif phase == "arlo_teaching":
        return base + f"""
This is a personalized teaching phase.
Block goal: {description}
The student said: \"{user_input}\"
Tutor them interactively — explain, quiz gently, and offer clarity.
"""
    else:
        return base + f"""
Student said: \"{user_input}\"
Reply helpfully based on input and context.
"""

def call_gpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return "Sorry, I had trouble generating a response."

@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_handler(request: Request, data: ChatbotInput):
    logger.info("Chatbot request received")
    user_id = extract_user_id(request, data)
    context = get_context_slice(user_id)
    prompt = build_prompt(data, context)
    logger.debug(f"Prompt built:\n{prompt}")
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

@router.post("/chatbot/save")
def save_chat_context(payload: Dict[str, Any]):
    try:
        logger.info(f"Saving chatbot context: {payload}")
        response = requests.post(f"{CONTEXT_API}/api/context/update", json=payload)
        response.raise_for_status()
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return {"status": "error", "detail": str(e)}

app.include_router(router)
