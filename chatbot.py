# UPDATED CHATBOT MODULE WITH IN-MODULE CONTEXT CACHE AND BETTER STUDY BLOCK DESCRIPTION HANDLING

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import openai
import os
import logging
import requests
from datetime import datetime, timedelta

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
# In-memory Context Cache
# ---------------------------
context_cache: dict = {}
context_ttl = timedelta(minutes=5)

def get_cached_context(user_id: str) -> Dict[str, Any]:
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return cached_value
    try:
        response = requests.get(f"{CONTEXT_API}/api/context/cache?user_id={user_id}", timeout=3)
        response.raise_for_status()
        raw = response.json()
        trimmed = {
            "current_topic": raw.get("current_topic"),
            "user_goals": raw.get("user_goals", [])[:2],
            "weak_areas": raw.get("weak_areas", [])[:2],
            "emphasized_facts": raw.get("emphasized_facts", [])[:2],
            "preferred_learning_styles": raw.get("preferred_learning_styles", [])[:1]
        }
        context_cache[user_id] = (now, trimmed)
        return trimmed
    except Exception as e:
        logger.warning(f"Context fetch failed: {e}")
        return {
            "current_topic": "general learning",
            "weak_areas": [],
            "user_goals": [],
            "emphasized_facts": [],
            "preferred_learning_styles": []
        }

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

def build_prompt(data: ChatbotInput, context: Dict[str, Any]) -> str:
    ctx = []
    if context.get("current_topic"):
        ctx.append(f"Current Topic: {context['current_topic']}")
    if context.get("weak_areas"):
        ctx.append(f"Weak Areas: {', '.join(context['weak_areas'])}")
    if context.get("emphasized_facts"):
        ctx.append(f"Important Facts: {', '.join(context['emphasized_facts'])}")

    # Add phase description from Lovable to stay on topic
    description = data.current_phase.description or ""
    phase_type = data.current_phase.phase
    technique = data.current_phase.tool

    history = "\n".join([f"{m['role']}: {m['content']}" for m in data.message_history[-2:]])

    prompt = (
        f"You are Arlo, an AI tutor helping a student with the current learning phase.\n"
        f"Technique: {technique}\n"
        f"Phase Type: {phase_type}\n"
        f"Description of task: {description}\n"
        f"{chr(10).join(ctx)}\n\n"
        f"Recent Conversation:\n{history}\n\n"
        f"Student input: \"{data.user_input.strip()}\"\n\n"
        "Your response should directly support the student's progress in this phase. Be concise, clear, and encouraging."
    )

    return prompt

def call_gpt(prompt: str) -> str:
    try:
        logger.info("Calling OpenAI GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300,
            request_timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return "Sorry, I had trouble generating a response."

# ---------------------------
# Main Route
# ---------------------------
@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_handler(request: Request, data: ChatbotInput):
    logger.info("Chatbot request received")
    try:
        user_id = extract_user_id(request, data)
        context = get_cached_context(user_id)
        prompt = build_prompt(data, context)
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
    except Exception as e:
        logger.error(f"Chatbot handler failed: {e}")
        raise HTTPException(status_code=500, detail="Chatbot failed to respond")

# ---------------------------
# Context Save Endpoint
# ---------------------------
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

# ---------------------------
# Include in App
# ---------------------------
app.include_router(router)
