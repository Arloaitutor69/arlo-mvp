# UPDATED BLURTING MODULE WITH IN-MODULE CONTEXT CACHE AND EXERCISE GENERATOR

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json
import requests
import time
from datetime import datetime, timedelta
from threading import Thread

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------- CONTEXT CACHE -------------------
context_cache: dict = {}
context_ttl = timedelta(minutes=5)

if os.getenv("ENV") == "dev":
    CONTEXT_BASE = "http://localhost:10000"
else:
    CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

def get_cached_context(user_id: str):
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return {"cached": True, "stale": False, "context": cached_value}
    try:
        response = requests.get(f"{CONTEXT_BASE}/api/context/cache?user_id={user_id}", timeout=5)
        response.raise_for_status()
        context = response.json()
        context_cache[user_id] = (now, context)
        return {"cached": False, "stale": False, "context": context}
    except Exception as e:
        print("❌ Failed to fetch cached context:", e)
        return {"cached": False, "stale": True, "context": None, "error": str(e)}

# ------------------- MODELS -------------------
class BlurtingRequest(BaseModel):
    topic: str
    content_summary: Optional[str] = None
    blurted_response: str
    context_prompt: Optional[str] = None
    user_id: Optional[str] = None

class BlurtingResponse(BaseModel):
    feedback: str
    missed_concepts: List[str]
    context_alignment: str

class BlurtingExerciseRequest(BaseModel):
    topic: str
    teaching_block: str
    user_id: Optional[str] = None

class BlurtingExerciseResponse(BaseModel):
    exercise_1: dict
    exercise_2: dict

# ------------------- USER ID EXTRACTION -------------------
def extract_user_id(request: Request, data) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif hasattr(data, 'user_id') and data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# ------------------- CONTEXT POSTING -------------------
def post_learning_event_to_context(user_id: str, topic: str, missed_concepts: List[str], feedback: str):
    payload = {
        "source": "blurting",
        "user_id": user_id,
        "current_topic": topic,
        "weak_areas": missed_concepts[:3],
        "review_queue": missed_concepts[:3],
        "learning_event": {
            "concept": topic,
            "phase": "blurting",
            "confidence": 2 if missed_concepts else 4,
            "depth": "shallow" if missed_concepts else "medium",
            "source_summary": feedback[:150],
            "repetition_count": 1,
            "review_scheduled": True
        }
    }
    try:
        start = time.time()
        res = requests.post(f"{CONTEXT_BASE}/api/context/update", json=payload, timeout=45)
        print("📦 Supabase context post response:", res.status_code, res.text)
        print("⏱️ Supabase log took", time.time() - start)
        res.raise_for_status()
        print("✅ Logged blurting to context.")
    except Exception as e:
        print(f"❌ Failed to log context: {e}")

def safe_context_log(user_id, topic, missed, feedback):
    try:
        post_learning_event_to_context(user_id, topic, missed, feedback)
    except Exception as e:
        print(f"⚠️ Background context log failed: {e}")

# ------------------- PROMPT BUILDER -------------------
def generate_blurting_prompt(topic: str, content_summary: Optional[str], blurted_response: str, context_prompt: Optional[str]) -> str:
    summary_block = f"\nSummary of key concepts:\n{content_summary}" if content_summary else ""
    context_block = f"\nAdditional context for evaluation:\n{context_prompt}" if context_prompt else ""
    return (
        f"You're an educational coach helping a student review their memory of the topic: \"{topic}\"."
        f"{summary_block}"
        f"The student wrote (trimmed if long):\n\"\"\"\n{blurted_response[:500]}\n\"\"\"\n\n"
        "Evaluate their explanation. Return a JSON object with:\n"
        "- \"feedback\": a paragraph highlighting what they did well and gently pointing out what was missing.\n"
        "- \"missed_concepts\": a list of key ideas or facts they forgot or explained poorly.\n"
        "- \"context_alignment\": a short sentence describing how well their answer aligns with the context prompt or learning goal, if provided.\n\n"
        "Only return valid JSON."
    )

# ------------------- MAIN ENDPOINT -------------------
@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: Request, data: BlurtingRequest):
    try:
        user_id = extract_user_id(request, data)

        if not data.context_prompt:
            result = get_cached_context(user_id)
            if result["context"]:
                data.context_prompt = json.dumps(result["context"].get("weak_areas", []))

        prompt = generate_blurting_prompt(
            data.topic,
            data.content_summary,
            data.blurted_response,
            data.context_prompt
        )

        start = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300,
            request_timeout=20
        )
        end = time.time()
        print(f"⏱️ GPT call took {end - start:.2f} seconds")

        content = response.choices[0].message.content.strip()
        print("🧠 GPT raw content:", content)
        parsed = json.loads(content)

        Thread(target=safe_context_log, args=(
            user_id,
            data.topic,
            parsed["missed_concepts"],
            parsed["feedback"]
        )).start()

        return BlurtingResponse(**parsed)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse GPT response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- BLURTING EXERCISE GENERATOR -------------------
@router.post("/blurting/exercises", response_model=BlurtingExerciseResponse)
async def generate_blurting_exercises(request: Request, data: BlurtingExerciseRequest):
    user_id = extract_user_id(request, data)
    context = get_cached_context(user_id).get("context") or {}

    prompt = (
        f"You are an AI tutor designing two blurting-style exercises to help a student consolidate knowledge on the topic: \"{data.topic}\".\n"
        f"They just studied this content:\n\"{data.teaching_block}\"\n\n"
        f"Your goal is to target parts of the content that involve a lot of memorization — lists, definitions, processes, detailed facts — not broad concepts.\n\n"
        f"Each exercise should instruct the student to recall as much as possible from memory.\n"
        f"Return JSON format:\n{{\n  'exercise_1': {{'prompt': ..., 'focus': ...}},\n  'exercise_2': {{'prompt': ..., 'focus': ...}}\n}}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful AI tutor."}, {"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=500,
            request_timeout=20
        )

        parsed = json.loads(response.choices[0].message.content.strip())
        return BlurtingExerciseResponse(
            exercise_1=parsed["exercise_1"],
            exercise_2=parsed["exercise_2"]
        )
    except Exception as e:
        print("❌ Exercise generation failed:", str(e))
        raise HTTPException(status_code=500, detail="Failed to generate blurting exercises.")
