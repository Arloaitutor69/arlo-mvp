from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
import uuid
import requests

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# --- Input model from the frontend or planner ---
class FlashcardRequest(BaseModel):
    topic: str
    content: Optional[str] = ""
    difficulty: Optional[str] = "medium"
    count: Optional[int] = 10
    format: Optional[str] = "Q&A"

# --- Output schema for flashcard items ---
class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str

# --- Helper to fetch context slice from context manager ---
def fetch_context_slice():
    try:
        response = requests.get(f"{CONTEXT_BASE_URL}/api/context/slice")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print("❌ Context slice fetch failed:", e)
    return {}

# --- Helper to post context update ---
def post_context_update(payload: dict):
    try:
        requests.post(f"{CONTEXT_BASE_URL}/api/context/update", json=payload)
    except Exception as e:
        print("❌ Context update failed:", e)

@router.post("/flashcards")
def generate_flashcards(data: FlashcardRequest):
    context = fetch_context_slice()

    personalization = ""
    if context:
        personalization = f"""
Session topic: {context.get('current_topic', data.topic)}
Emphasize: {', '.join(context.get('emphasized_facts', [])) or 'N/A'}
Weak areas: {', '.join(context.get('weak_areas', [])) or 'N/A'}
Goals: {', '.join(context.get('user_goals', [])) or 'N/A'}
Review queue: {', '.join(context.get('review_queue', [])) or 'N/A'}
"""

    # --- Optimized, concise prompt using GPT-4 ---
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
            model="gpt-4",
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

    # ✅ Log session context for review and tracking
    post_context_update({
        "source": "flashcards",
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
