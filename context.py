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
from fastapi import Request


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
    user_id: Optional[str] = None  # ← Add this line
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


def synthesize_context_gpt(user_id: str) -> dict:
    # 1. Get recent context logs for this user only
    logs_raw = get_supabase() \
        .table("context_log") \
        .select("learning_event, current_topic, user_goals, source, weak_areas, emphasized_facts, preferred_learning_styles, review_queue") \
        .eq("user_id", user_id) \
        .order("id", desc=True) \
        .limit(10) \
        .execute().data[::-1]  # chronological order

    # 2. Build merged raw input
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

    # Pull best available non-null metadata from recent entries
    def first_valid(field):
        for e in reversed(logs_raw):  # prioritize newer
            val = e.get(field)
            if isinstance(val, list) and val:
                return val
            elif isinstance(val, str) and val.strip():
                return val
        return [] if field != "current_topic" else None

    context_fields = {
        "current_topic": first_valid("current_topic"),
        "user_goals": first_valid("user_goals"),
        "weak_areas": first_valid("weak_areas"),
        "emphasized_facts": first_valid("emphasized_facts"),
        "preferred_learning_styles": first_valid("preferred_learning_styles"),
        "review_queue": first_valid("review_queue")
    }

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
        context_json = extract_and_parse_json(raw_content)

        # 3. Insert best metadata into final state
        context_json.update(context_fields)

        # 4. Save into per-user context_state using UPSERT
        get_supabase().table("context_state").upsert({
            "user_id": user_id,
            "context": json.dumps(context_json)
        }, on_conflict=["user_id"]).execute()

        return context_json

    except Exception as e:
        print("❌ GPT synthesis failed:", e)
        raise HTTPException(status_code=500, detail=f"Failed to parse GPT output: {e}")
# ------------------------------
# Routes
# ------------------------------
@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    entry = update.dict(exclude={"trigger_synthesis", "user_id"})
    entry["timestamp"] = datetime.utcnow().isoformat()

    # --- Robust user ID extraction ---
    user_info = getattr(request.state, "user", None)

    if user_info and "sub" in user_info:
        user_id = user_info["sub"]
    elif update.user_id:
        user_id = update.user_id
    elif update.source and update.source.startswith("user:"):
        user_id = update.source.replace("user:", "")
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing or invalid user ID. Include 'user_id' or use 'source=\"user:<uuid>\"'."
        )

    entry["user_id"] = user_id

    # --- Debugging logs ---
    try:
        print("🔍 DEBUG: source =", update.source)
        print("🔍 DEBUG: user_id =", user_id)
        print("🔍 DEBUG: full context entry =", json.dumps(entry, indent=2))
    except Exception as log_err:
        print("⚠️ Logging error:", log_err)

    # --- Save raw log entry ---
    try:
        get_supabase().table("context_log").insert(entry).execute()
    except Exception as db_err:
        print("❌ DB Insert Error:", db_err)
        raise HTTPException(status_code=500, detail="Failed to save context")

    # --- Optionally synthesize ---
    if should_trigger_synthesis(update):
        try:
            synthesized = synthesize_context_gpt(user_id=user_id)

            # Save per-user synthesized context
            get_supabase().table("context_state").upsert({
                "user_id": user_id,
                "context": json.dumps(synthesized)
            }, on_conflict=["user_id"]).execute()

            # Purge older logs for this user beyond most recent 5
            all_logs = get_supabase().table("context_log") \
                .select("id") \
                .eq("user_id", user_id) \
                .order("id", desc=True) \
                .execute().data

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

        # 2️⃣ Delete existing context_state row for this user
        delete_ctx_url = f"{supabase_url}/rest/v1/context_state?user_id=eq.{user_id}"
        requests.delete(delete_ctx_url, headers=headers)

        # 3️⃣ Insert clean blank context state for this user
        reset_context = {
            "user_id": user_id,
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

        reset_res = requests.post(
            f"{supabase_url}/rest/v1/context_state",
            json=reset_context,
            headers=headers
        )
        reset_res.raise_for_status()

        return {"status": "context fully wiped"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

def is_stale(timestamp: str, ttl_seconds: int = 120) -> bool:
    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return datetime.now(timezone.utc) - dt > timedelta(seconds=ttl_seconds)

@router.get("/context/cache")
def get_cached_context(user_id: str):
    # Load Supabase settings dynamically
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE")

    if not supabase_url or not supabase_key:
        raise HTTPException(status_code=500, detail="Missing SUPABASE_URL or SERVICE_ROLE key")

    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }

    # 1. Try fetching from Supabase cache
    url = f"{supabase_url}/rest/v1/context_cache?user_id=eq.{user_id}&select=*"
    resp = requests.get(url, headers=headers)

    if resp.status_code == 200 and resp.json():
        row = resp.json()[0]
        cached = json.loads(row["cached_json"])
        timestamp = row["last_updated"]
        if not is_stale(timestamp):
            return {"cached": True, "stale": False, "context": cached}

    # 2. Fetch fresh context from context state
    context_api_base = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")
    state_url = f"{context_api_base}/api/context/state?user_id={user_id}"
    state_resp = requests.get(state_url)
    if state_resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch fresh context")

    context = state_resp.json()

    # 3. Store fresh context back into Supabase
    upsert_payload = [{
        "user_id": user_id,
        "cached_json": json.dumps(context),
        "last_updated": datetime.now(timezone.utc).isoformat()
    }]
    upsert_url = f"{supabase_url}/rest/v1/context_cache"
    upsert_resp = requests.post(
        upsert_url,
        headers={**headers, "Prefer": "resolution=merge-duplicates"},
        json=upsert_payload
    )

    if upsert_resp.status_code not in [200, 201]:
        raise HTTPException(status_code=500, detail="Failed to save context cache")

    return {"cached": False, "stale": True, "context": context}

@router.get("/context/logs/recent")
def get_recent_logs(user_id: str):
    return query_context_log_table(user_id, limit=5)

@router.get("/context/state")
def get_context_state(user_id: str):
    try:
        result = get_supabase() \
            .table("context_state") \
            .select("context") \
            .eq("user_id", user_id) \
            .single() \
            .execute()

        ctx_raw = result.data.get("context") if result.data else None
        if not ctx_raw:
            raise ValueError("No context found for user")

        return json.loads(ctx_raw)

    except Exception as e:
        print("❌ Error fetching context state:", e)
        raise HTTPException(status_code=404, detail=f"Context state not found: {e}")

@router.get("/context/slice")
async def get_context_slice(request: Request):
    try:
        # --- Extract user ID ---
        user_info = getattr(request.state, "user", None)
        user_id = (
            user_info["sub"]
            if user_info and "sub" in user_info else
            request.headers.get("X-User-ID") or
            request.query_params.get("user_id")
        )

        if not user_id:
            raise ValueError("No user_id found in auth, header, or query")

        # --- Fetch context for user ---
        res = get_supabase().table("context_state").select("context").eq("user_id", user_id).single().execute()
        ctx_raw = res.data.get("context") if res.data else None
        if not ctx_raw:
            raise ValueError(f"No context found for user_id={user_id}")

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
