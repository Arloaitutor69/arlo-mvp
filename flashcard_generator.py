# ✅ Fully patched flashcard module with context-aware learning_event logging

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
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "http://localhost:10000")

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

@router.post("/api/flashcards")
def generate_flashcards(data: FlashcardRequest):
    context = fetch_context_slice()

    # Build GPT prompt using context slice to personalize
    personalization = ""
    if context:
        current_topic = context.get("current_topic", "")
        emphasized = ", ".join(context.get("emphasized_facts", []))
        weak_areas = ", ".join(context.get("weak_areas", []))
        user_goals = ", ".join(context.get("user_goals", []))
        review_queue = ", ".join(context.get("review_queue", []))

        personalization = f"""
Current session topic: {current_topic or data.topic}
Emphasize these facts: {emphasized or 'N/A'}
Prioritize these weak areas: {weak_areas or 'N/A'}
Tailor to user goals: {user_goals or 'N/A'}
Optionally include 1–2 review cards on: {review_queue or 'none'}
"""

    prompt = f"""
You are a flashcard tutor generating study cards.

Topic: "{data.topic}"
Notes: "{data.content or 'Use general knowledge if no notes provided.'}"

Difficulty: {data.difficulty}
Format: {data.format}

{personalization}

Use only one of these formats:
- "Q&A" → Basic question/answer
- "fill-in-the-blank" → Sentence with a missing term
- "multiple-choice" → (Not supported yet — just return Q&A for now)

Return ONLY a valid JSON array of objects like:
[
  {{ "question": "What is ...?", "answer": "..." }},
  ...
]

Do not include explanations, headers, or any other text — just the JSON.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()

        # Handle GPT formatting edge case
        if raw.startswith("```"):
            raw = "\n".join(raw.strip().splitlines()[1:-1])

        print("🧐 Raw GPT output:\n", raw)

        try:
            cards = json.loads(raw)
        except Exception as parse_error:
            print("❌ JSON parsing failed:\n", parse_error)
            print("🔎 GPT raw output again:", raw)
            raise HTTPException(status_code=500, detail="GPT returned bad JSON format.")

    except Exception as e:
        print("⚠️ GPT API Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate flashcards.")

    flashcards = []
    questions_summary = []

    for item in cards[:data.count]:
        flashcards.append(FlashcardItem(
            id=f"card_{uuid.uuid4().hex[:6]}",
            front=item.get("question", "No question."),
            back=item.get("answer", "No answer."),
            difficulty=data.difficulty,
            category=data.topic
        ))
        questions_summary.append(item.get("question", ""))

    # ✅ Post to context manager with learning_event and metadata
    post_context_update({
        "source": "flashcards",
        "phase": "flashcards",
        "event_type": "generation",
        "learning_event": {
            "concept": data.topic,
            "phase": "flashcards",
            "confidence": 0.5,
            "depth": "shallow",
            "source_summary": "; ".join(questions_summary),
            "repetition_count": 1,
            "review_scheduled": False
        },
        "data": {
            "topic": data.topic,
            "difficulty": data.difficulty,
            "flashcard_count": len(flashcards)
        }
    })

    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }
