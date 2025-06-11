from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
import uuid
import json
import openai
import os
from supabase import create_client, Client

# ------------------------------
# Supabase lazy initialization
# ------------------------------
supabase: Optional[Client] = None

def get_supabase() -> Client:
    global supabase
    if not supabase:
        url = os.getenv("SUPABASE_URL") or os.getenv("SUPABASE_UR")
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE not set")
        supabase = create_client(url, key)
    return supabase

# ------------------------------
# Router
# ------------------------------
router = APIRouter()

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
# Helper Functions
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
    res = get_supabase().table("context_log").select("source").order("id", desc=True).limit(5).execute()
    recent_entries = res.data if res.data else []
    if new_entry.feedback_flag:
        return True
    if len(recent_entries) >= 3:
        sources = {entry["source"] for entry in recent_entries}
        if len(sources) >= 2:
            return True
    return False

def synthesize_context_gpt() -> dict:
    logs = get_supabase().table("context_log").select("*").order("id").execute().data

    prompt = f"""
You are ARLO's memory engine. Read the raw study logs below and return ONLY valid JSON.
Do not include markdown, explanation, or formatting.
Your response must exactly match this structure:

{{
  "current_topic": string or null,
  "user_goals": [string],
  "weak_areas": [string],
  "emphasized_facts": [string],
  "preferred_learning_styles": [string],
  "review_queue": [string],
  "learning_history": [
    {{
      "concept": string,
      "phase": string,
      "confidence": float,
      "depth": "shallow" | "intermediate" | "deep",
      "source_summary": string,
      "repetition_count": int,
      "review_scheduled": boolean
    }}
  ]
}}

Raw Logs:
{json.dumps(logs, indent=2)}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a context synthesis engine for an educational AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )

    raw_content = response.choices[0].message["content"]
    try:
        parsed = json.loads(raw_content)
        return parsed
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Failed to parse GPT output:\n{raw_content}")

# ------------------------------
# Routes
# ------------------------------
@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    entry = update.dict()
    entry["timestamp"] = datetime.utcnow().isoformat()

    # Attempt user extraction but skip enforcement
    user_info = getattr(request.state, "user", None)
    # Use a fixed dummy UUID in dev mode if user is not authenticated
    entry["user_id"] = (
        user_info["sub"]
        if user_info and "sub" in user_info
        else "00000000-0000-0000-0000-000000000000"
    )

    get_supabase().table("context_log").insert(entry).execute()

    if should_trigger_synthesis(update):
        synthesized = synthesize_context_gpt()
        get_supabase().table("context_state").delete().neq("id", 0).execute()  # Clear previous
        get_supabase().table("context_state").insert({"id": 1, "context": json.dumps(synthesized)}).execute()
        return {"status": "ok", "synthesized": True}

    return {"status": "ok", "synthesized": False}

@router.get("/context/current")
async def get_full_context():
    res = get_supabase().table("context_state").select("context").eq("id", 1).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Context not yet synthesized.")
    return json.loads(res.data["context"])

@router.get("/context/slice")
async def get_context_slice():
    res = get_supabase().table("context_state").select("context").eq("id", 1).single().execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Context not yet synthesized.")
    ctx = json.loads(res.data["context"])
    return {
        "current_topic": ctx.get("current_topic"),
        "user_goals": ctx.get("user_goals"),
        "weak_areas": ctx.get("weak_areas"),
        "emphasized_facts": ctx.get("emphasized_facts"),
        "preferred_learning_styles": ctx.get("preferred_learning_styles"),
        "review_queue": ctx.get("review_queue")
    }
