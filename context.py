from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime
import json
import openai
import os
from supabase import create_client, Client
import re
import requests

# ------------------------------
# Supabase lazy initialization
# ------------------------------
supabase: Optional[Client] = None

def get_supabase() -> Client:
    global supabase
    if not supabase:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE")
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
    trigger_synthesis: Optional[bool] = False

class ContextResetRequest(BaseModel):
    user_id: str

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

def should_trigger_synthesis(update: ContextUpdate) -> bool:
    if update.trigger_synthesis:
        return True
    res = get_supabase().table("context_log").select("source").order("id", desc=True).limit(5).execute()
    recent_entries = res.data if res.data else []
    if update.feedback_flag:
        return True
    if len(recent_entries) >= 3:
        sources = {entry["source"] for entry in recent_entries}
        if len(sources) >= 2:
            return True
    return False


def extract_and_parse_json(text: str) -> dict:
    """
    Extracts the first valid JSON object from a text blob and parses it safely.
    Assumes object starts with the first '{' and ends with the matching '}'.
    """
    try:
        start = text.index('{')
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        else:
            raise ValueError("Mismatched braces in GPT output")

        json_str = text[start:end]

        # Sanitize common GPT errors: remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        return json.loads(json_str)

    except Exception as e:
        print("❌ Robust JSON parse failed:", e)
        print("🔴 Raw GPT output:\n", text)
        return {"status": "error", "synthesized": False}


def synthesize_context_gpt() -> dict:
    logs_raw = get_supabase().table("context_log").select("learning_event, current_topic, source").order("id", desc=True).limit(10).execute().data[::-1]

    flattened_logs = []
    for entry in logs_raw:
        le = entry.get("learning_event") or {}
        flattened_logs.append({
            "concept": le.get("concept", ""),
            "phase": le.get("phase", ""),
            "confidence": le.get("confidence", 0.5),
            "depth": le.get("depth", "intermediate"),
            "source_summary": le.get("source_summary", ""),
            "repetition_count": le.get("repetition_count", 0),
            "review_scheduled": le.get("review_scheduled", False)
        })

    prompt = f"""
You are ARLO's memory engine. Read the raw study logs below and return ONLY valid JSON.
Return a single object matching this structure:

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

DO NOT include markdown. No commentary. No explanation. No lists. Only a single valid JSON object. No null values — replace with defaults.

Raw Logs:
{json.dumps(flattened_logs, indent=2)}
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a context synthesis engine for an educational AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        raw_content = response.choices[0].message["content"].strip()
        return extract_and_parse_json(raw_content)

    except Exception as e:
        print("❌ GPT synthesis failed:", e)
        raise HTTPException(status_code=500, detail=f"Failed to parse GPT output: {e}")
# ------------------------------
# Routes
# ------------------------------
@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    entry = update.dict(exclude={"trigger_synthesis"})
    entry["timestamp"] = datetime.utcnow().isoformat()

    user_info = getattr(request.state, "user", None)
    
    if user_info and "sub" in user_info:
        user_id = user_info["sub"]
    elif update.source and update.source.startswith("user:"):
        user_id = update.source.replace("user:", "")
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid user ID. Ensure frontend passes a valid auth token or user:UUID in 'source'."
        )

    entry["user_id"] = user_id

    try:
        print("\U0001f6a8 DEBUG: source =", update.source)
        print("\U0001f6a8 DEBUG: user_id =", user_id)
        print("\U0001f6a8 DEBUG: full context entry =", json.dumps(entry, indent=2))
    except Exception as log_err:
        print("⚠️ Logging error:", log_err)

    try:
        get_supabase().table("context_log").insert(entry).execute()
    except Exception as db_err:
        print("❌ DB Insert Error:", db_err)
        raise HTTPException(status_code=500, detail="Failed to save context")

    if should_trigger_synthesis(update):
        try:
            synthesized = synthesize_context_gpt()
            get_supabase().table("context_state").delete().neq("id", 0).execute()
            get_supabase().table("context_state").insert({
                "id": 1,
                "context": json.dumps(synthesized)
            }).execute()

            # Optional cleanup: purge old logs beyond the last 5
            all_logs = get_supabase().table("context_log").select("id").order("id", desc=True).execute().data
            ids_to_keep = {entry["id"] for entry in all_logs[:5]}
            for entry in all_logs[5:]:
                if entry["id"] not in ids_to_keep:
                    get_supabase().table("context_log").delete().eq("id", entry["id"]).execute()

            return {"status": "ok", "synthesized": True}
        except Exception as synth_err:
            print("❌ Synthesis Error:", synth_err)
            return {"status": "error", "synthesized": False}

    return {"status": "ok", "synthesized": False}

@router.post("/context/reset")
def reset_context_state(request: ContextResetRequest):
    user_id = request.user_id

    try:
        # Load Supabase credentials
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE")

        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Missing Supabase env variables")

        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }

        # 1️⃣ Delete all context_log rows for the user
        delete_logs_url = f"{supabase_url}/rest/v1/context_log?user_id=eq.{user_id}"
        delete_res = requests.delete(delete_logs_url, headers=headers)
        delete_res.raise_for_status()

        # 2️⃣ Reset the context_state row to clean values
        reset_context = {
            "id": 1,
            "context": json.dumps({
                "current_topic": None,
                "user_goals": [],
                "preferred_learning_styles": [],
                "weak_areas": [],
                "emphasized_facts": [],
                "review_queue": [],
                "learning_history": []
            })
        }

        # Delete existing context_state
        requests.delete(f"{supabase_url}/rest/v1/context_state?id=eq.1", headers=headers)

        # Insert fresh reset context
        reset_res = requests.post(
            f"{supabase_url}/rest/v1/context_state",
            json=reset_context,
            headers=headers
        )
        reset_res.raise_for_status()

        return {"status": "context fully wiped"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@router.get("/context/logs/recent")
def get_recent_logs(user_id: str):
    return query_context_log_table(user_id, limit=5)

@router.get("/context/slice")
async def get_context_slice():
    try:
        res = get_supabase().table("context_state").select("context").eq("id", 1).single().execute()
        ctx_raw = res.data.get("context") if res.data else None
        if not ctx_raw:
            raise ValueError("No context data found")
        ctx = json.loads(ctx_raw)
    except Exception as e:
        print("⚠️ Context fallback triggered:", e)
        ctx = {
            "current_topic": None,
            "user_goals": [],
            "weak_areas": [],
            "emphasized_facts": [],
            "preferred_learning_styles": [],
            "review_queue": []
        }

    return {
        "current_topic": ctx.get("current_topic"),
        "user_goals": ctx.get("user_goals", []),
        "weak_areas": ctx.get("weak_areas", []),
        "emphasized_facts": ctx.get("emphasized_facts", []),
        "preferred_learning_styles": ctx.get("preferred_learning_styles", []),
        "review_queue": ctx.get("review_queue", [])
    }
