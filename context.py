from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import uuid
import json
import openai

router = APIRouter()

# ------------------------------
# In-memory storage (can be replaced by Supabase later)
# ------------------------------

raw_log = []
latest_context = {}

# ------------------------------
# Pydantic Models
# ------------------------------

class LearningEvent(BaseModel):
    concept: str
    phase: str
    confidence: Optional[float] = 0.5
    depth: Optional[Literal['shallow', 'intermediate', 'deep']] = 'shallow'
    source_summary: Optional[str] = None
    repetition_count: Optional[int] = 1
    review_scheduled: Optional[bool] = False

class ContextUpdate(BaseModel):
    current_topic: Optional[str] = None
    user_goals: Optional[List[str]] = None
    preferred_learning_styles: Optional[List[str]] = None
    weak_areas: Optional[List[str]] = None
    emphasized_facts: Optional[List[str]] = None
    review_queue: Optional[List[str]] = None
    learning_event: Optional[LearningEvent] = None
    source: str
    feedback_flag: Optional[bool] = False

# ------------------------------
# Utilities
# ------------------------------

def score_source(source: str) -> int:
    priority = {
        'session_planner': 100,
        'chatbot': 80,
        'flashcards': 70,
        'quiz': 70,
        'feynman': 70,
        'blurting': 70,
        'review_sheet': 60,
        'chatbot_flag': 90
    }
    return priority.get(source, 50)

def should_trigger_synthesis(new_entry: ContextUpdate) -> bool:
    if new_entry.feedback_flag:
        return True
    recent_entries = raw_log[-5:]
    if len(recent_entries) >= 3:
        sources = {entry.source for entry in recent_entries}
        if len(sources) >= 2:
            return True
    return False

def synthesize_context() -> dict:
    # This could be replaced by a call to GPT. For now we simulate logic.
    context = {
        "current_topic": None,
        "user_goals": [],
        "preferred_learning_styles": [],
        "weak_areas": [],
        "emphasized_facts": [],
        "review_queue": [],
        "learning_history": []
    }
    for entry in raw_log:
        data = entry.dict()
        if data.get("current_topic") and not context["current_topic"]:
            context["current_topic"] = data["current_topic"]
        context["user_goals"] += data.get("user_goals", [])
        context["preferred_learning_styles"] += data.get("preferred_learning_styles", [])
        context["weak_areas"] += data.get("weak_areas", [])
        context["emphasized_facts"] += data.get("emphasized_facts", [])
        context["review_queue"] += data.get("review_queue", [])
        if data.get("learning_event"):
            context["learning_history"].append(data["learning_event"].dict())
    # Deduplicate
    for key in ["user_goals", "preferred_learning_styles", "weak_areas", "emphasized_facts", "review_queue"]:
        context[key] = list(set(context[key]))
    return context

# ------------------------------
# Routes
# ------------------------------

@router.post("/context/update")
async def update_context(update: ContextUpdate):
    raw_log.append(update)
    if should_trigger_synthesis(update):
        global latest_context
        latest_context = synthesize_context()
    return {"status": "ok", "synthesized": should_trigger_synthesis(update)}

@router.get("/context/current")
async def get_full_context():
    if not latest_context:
        raise HTTPException(status_code=404, detail="Context not yet synthesized.")
    return latest_context

@router.get("/context/slice")
async def get_context_slice():
    if not latest_context:
        raise HTTPException(status_code=404, detail="Context not yet synthesized.")
    return {
        "current_topic": latest_context.get("current_topic"),
        "user_goals": latest_context.get("user_goals"),
        "weak_areas": latest_context.get("weak_areas"),
        "emphasized_facts": latest_context.get("emphasized_facts"),
        "preferred_learning_styles": latest_context.get("preferred_learning_styles"),
        "review_queue": latest_context.get("review_queue")
    }
