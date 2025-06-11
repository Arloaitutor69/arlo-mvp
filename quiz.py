# ✅ Clean and prefix-safe quiz module (mounted as /api/quiz)
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import os
import json
import openai
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# --- Models ---
class QuizRequest(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    question_count: int
    question_types: List[Literal["multiple_choice", "true_false", "fill_in_blank"]]

class QuizQuestion(BaseModel):
    id: int
    type: Literal["multiple_choice", "true_false", "fill_in_blank"]
    question: str
    options: Union[List[str], None] = None
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]

# --- Context Helpers ---
def fetch_context():
    try:
        print("📥 Fetching context slice...")
        res = requests.get(f"{CONTEXT_API}/api/context/slice", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return {}

def log_learning_event(topic, summary, count):
    try:
        payload = {
            "source": "quiz",
            "phase": "quiz",
            "event_type": "generation",
            "learning_event": {
                "concept": topic,
                "phase": "quiz",
                "confidence": 0.5,
                "depth": "shallow",
                "source_summary": summary,
                "repetition_count": 1,
                "review_scheduled": False
            },
            "data": {
                "topic": topic,
                "question_count": count
            }
        }
        res = requests.post(f"{CONTEXT_API}/api/context/update", json=payload, timeout=10)
        print("📤 Context updated:", res.status_code)
    except Exception as e:
        print("❌ Failed to log learning event:", e)

# --- GPT Helper ---
def generate_questions(topic, difficulty, count, types, context):
    system_msg = "You are a quiz tutor who writes clear, factual questions in JSON."

    current_topic = context.get("current_topic", "")
    weak_areas = ", ".join(context.get("weak_areas", []))
    emphasized_facts = ", ".join(context.get("emphasized_facts", []))
    user_goals = ", ".join(context.get("user_goals", []))
    review_queue = ", ".join(context.get("review_queue", []))

    user_msg = f"""
Generate {count} quiz questions on the topic: \"{topic}\" (current context topic: \"{current_topic}\").
Difficulty: \"{difficulty}\". Allowed types: {types}.

Personalize:
- Weak areas: {weak_areas or 'none'}
- Emphasized facts: {emphasized_facts or 'none'}
- User goals: {user_goals or 'none'}
- Include 1–2 questions from review queue: {review_queue or 'none'}

Return ONLY a valid JSON array of objects like:
[
  {{
    "id": 1,
    "type": "multiple_choice",
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "...",
    "explanation": "..."
  }}
]
No extra text. No markdown.
"""

    try:
        print("🧠 Sending request to GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.6,
            max_tokens=1200
        )

        raw = response["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1])

        parsed = json.loads(raw)
        return [QuizQuestion(**q) for q in parsed]

    except Exception as e:
        print("❌ GPT generation failed:", e)
        raise HTTPException(status_code=500, detail="GPT quiz generation failed")

# --- Routes (prefix-safe) ---
@router.get("/test")
def quiz_health_check():
    return {"status": "quiz router live"}

@router.get("/test-log")
def test_learning_log():
    payload = {
        "source": "quiz",
        "phase": "quiz",
        "event_type": "generation",
        "learning_event": {
            "concept": "Test Concept",
            "phase": "quiz",
            "confidence": 0.5,
            "depth": "shallow",
            "source_summary": "Sample quiz test for logging verification.",
            "repetition_count": 1,
            "review_scheduled": False
        },
        "data": {
            "topic": "Test Concept",
            "question_count": 1
        }
    }

    try:
        res = requests.post(
            f"{CONTEXT_API}/api/context/update",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        return {
            "status": "posted",
            "code": res.status_code,
            "text": res.text
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
