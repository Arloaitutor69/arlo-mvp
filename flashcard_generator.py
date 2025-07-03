# UPDATED FLASHCARDS MODULE WITH SHARED IN-MODULE CONTEXT CACHE

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
import uuid
import requests
from datetime import datetime, timedelta

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# In-memory Context Cache
# -----------------------------
context_cache = {}
context_ttl = timedelta(minutes=5)

def get_cached_context(user_id: str):
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return cached_value
    try:
        res = requests.get(f"{CONTEXT_BASE_URL}/api/context/current?user_id={user_id}", timeout=5)
        res.raise_for_status()
        context = res.json()
        context_cache[user_id] = (now, context)
        return context
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return {}

# --- Input model from the frontend or planner ---
class FlashcardRequest(BaseModel):
    topic: str
    content: Optional[str] = ""
    difficulty: Optional[str] = "medium"
    count: Optional[int] = 10
    format: Optional[str] = "Q&A"
    user_id: Optional[str] = None

# --- Output schema for flashcard items ---
class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str

# --- Extract user_id ---
def extract_user_id(request: Request, data: FlashcardRequest) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# --- Helper to post context update ---
def post_context_update(payload: dict):
    try:
        requests.post(f"{CONTEXT_BASE_URL}/api/context/update", json=payload)
    except Exception as e:
        print("❌ Context update failed:", e)

@router.post("/flashcards")
def generate_flashcards(request: Request, data: FlashcardRequest):
    user_id = extract_user_id(request, data)
    context = get_cached_context(user_id)

    personalization = ""
    if context:
        personalization = f"""
Session topic: {context.get('current_topic', data.topic)}
Emphasize: {', '.join(context.get('emphasized_facts', [])) or 'N/A'}
Weak areas: {', '.join(context.get('weak_areas', [])) or 'N/A'}
Goals: {', '.join(context.get('user_goals', [])) or 'N/A'}
Review queue: {', '.join(context.get('review_queue', [])) or 'N/A'}
"""

    prompt = f"""
You are a flashcard-generating tutor.

Topic: {data.topic}
Notes: {data.content or 'Use general knowledge.'}

Create exactly {data.count} flashcards as a JSON array with objects like:
  {{ "question": "...", "answer": "..." }}

Requirements:
- Format: {data.format} (only Q&A supported for now)
- Difficulty: {data.difficulty}
- Include variety: facts, definitions, exceptions, how-to steps
- Prioritize clarity, helpfulness, and spaced retention
- Adapt based on: {personalization.strip()}

Return only the JSON array — no other text.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = "\n".join(raw.strip().splitlines()[1:-1])

        cards = json.loads(raw)

    except Exception as e:
        print("❌ GPT or parsing error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate flashcards.")

    flashcards = []
    questions_summary = []

    for item in cards[:data.count]:
        q = item.get("question", "No question.")
        a = item.get("answer", "No answer.")
        flashcards.append(FlashcardItem(
            id=f"card_{uuid.uuid4().hex[:6]}",
            front=q,
            back=a,
            difficulty=data.difficulty,
            category=data.topic
        ))
        questions_summary.append(q)

    post_context_update({
        "source": "flashcards",
        "user_id": user_id,
        "current_topic": data.topic,
        "learning_event": {
            "concept": data.topic,
            "phase": "flashcards",
            "confidence": 0.5,
            "depth": "shallow",
            "source_summary": "; ".join(questions_summary),
            "repetition_count": 1,
            "review_scheduled": False
        }
    })

    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }
