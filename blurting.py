from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json
import requests
import time
from threading import Thread

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set local or prod context API base
if os.getenv("ENV") == "dev":
    CONTEXT_BASE = "http://localhost:10000"
else:
    CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

# Models
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

# --- USER ID EXTRACTION ---
def extract_user_id(request: Request, data: BlurtingRequest) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# Context functions
def get_context_slice(user_id: str):
    try:
        res = requests.get(f"{CONTEXT_BASE}/api/context/slice?user_id={user_id}", timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return None

def safe_context_fetch(user_id: str, result_holder: dict):
    result_holder['data'] = get_context_slice(user_id)

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

# Prompt builder
def generate_blurting_prompt(topic: str, content_summary: Optional[str], blurted_response: str, context_prompt: Optional[str]) -> str:
    summary_block = f"\nSummary of key concepts:\n{content_summary}" if content_summary else ""
    context_block = f"\nAdditional context for evaluation:\n{context_prompt}" if context_prompt else ""
    return (
        f"You're an educational coach helping a student review their memory of the topic: \"{topic}\"."
        f"{summary_block}"
        f"{context_block}\n\n"
        f"The student wrote (trimmed if long):\n\"\"\"\n{blurted_response[:500]}\n\"\"\"\n\n"
        "Evaluate their explanation. Return a JSON object with:\n"
        "- \"feedback\": a paragraph highlighting what they did well and gently pointing out what was missing.\n"
        "- \"missed_concepts\": a list of key ideas or facts they forgot or explained poorly.\n"
        "- \"context_alignment\": a short sentence describing how well their answer aligns with the context prompt or learning goal, if provided.\n\n"
        "Only return valid JSON."
    )

# Endpoint
@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: Request, data: BlurtingRequest):
    try:
        user_id = extract_user_id(request, data)

        if not data.context_prompt:
            context_holder = {}
            fetch_thread = Thread(target=safe_context_fetch, args=(user_id, context_holder))
            fetch_thread.start()
            fetch_thread.join(timeout=3)
            context_data = context_holder.get('data')
            data.context_prompt = json.dumps(context_data.get("weak_areas", [])) if context_data else None

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
