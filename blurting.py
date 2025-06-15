from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json
import requests
import time

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

# Context functions
def get_context_slice():
    try:
        res = requests.get(f"{CONTEXT_BASE}/api/context/slice", timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return None

def post_learning_event_to_context(user_id: str, topic: str, missed_concepts: List[str], feedback: str):
    payload = {
        "source": f"user:{user_id}",
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
async def evaluate_blurting(request: BlurtingRequest):
    try:
        if not request.context_prompt:
            context_data = get_context_slice()
            request.context_prompt = json.dumps(context_data.get("weak_areas", [])) if context_data else None

        prompt = generate_blurting_prompt(
            request.topic,
            request.content_summary,
            request.blurted_response,
            request.context_prompt
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
        parsed = json.loads(content)

        if request.user_id:
            post_learning_event_to_context(
                request.user_id,
                request.topic,
                parsed["missed_concepts"],
                parsed["feedback"]
            )

        return BlurtingResponse(**parsed)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse GPT response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
