# UPDATED QUIZ MODULE WITH SHARED IN-MODULE CONTEXT CACHE

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Literal, Union, Optional
import uuid
import os
import json
import openai
import requests
from datetime import datetime, timedelta

openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

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
        res = requests.get(f"{CONTEXT_API}/api/context/cache?user_id={user_id}", timeout=5)
        res.raise_for_status()
        context = res.json()
        context_cache[user_id] = (now, context)
        return context
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return {}

# --- Models ---
class QuizRequest(BaseModel):
    topic: str
    difficulty: Optional[Literal["easy", "medium", "hard"]] = "medium"
    question_count: Optional[int] = 5
    question_types: Optional[List[Literal["multiple_choice", "true_false"]]] = None
    source: Optional[str] = None

class QuizQuestion(BaseModel):
    id: int
    type: Literal["multiple_choice", "true_false"]
    question: str
    options: Union[List[str], None] = None
    correct_answer: str
    explanation: str

class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]

# --- User ID Extraction ---
def extract_user_id(request: Request, req: QuizRequest) -> Optional[str]:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif req.source and req.source.startswith("user:"):
        return req.source.replace("user:", "")
    else:
        return None

# --- Log learning event ---
def log_learning_event(topic, summary, count, user_id: Optional[str] = None):
    try:
        payload = {
            "source": "quiz",
            "user_id": user_id,
            "current_topic": topic,
            "learning_event": {
                "concept": topic,
                "phase": "quiz",
                "confidence": 0.5,
                "depth": "shallow",
                "source_summary": summary,
                "repetition_count": 1,
                "review_scheduled": False
            }
        }
        res = requests.post(
            f"{CONTEXT_API}/api/context/update",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        print("📬 Context updated:", res.status_code)
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
Generate {count} quiz questions on the topic: \"{topic}\" (context: \"{current_topic}\").
Difficulty: \"{difficulty}\".
Allowed types: {types}.

Personalize:
- Weak areas: {weak_areas or 'none'}
- Emphasized facts: {emphasized_facts or 'none'}
- User goals: {user_goals or 'none'}
- Include 1–2 questions from review queue: {review_queue or 'none'}

Return ONLY a valid JSON array like:
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
No markdown, no preamble.
"""

    try:
        print("🧐 Sending request to GPT...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
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

        for q in parsed:
            if isinstance(q.get("correct_answer"), bool):
                q["correct_answer"] = str(q["correct_answer"])

        return [QuizQuestion(**q) for q in parsed]

    except Exception as e:
        print("❌ GPT generation failed:", e)
        raise HTTPException(status_code=500, detail="GPT quiz generation failed")

# --- Routes ---
@router.get("/test")
def quiz_health_check():
    return {"status": "quiz router live"}

@router.post("/generate", response_model=QuizResponse)
async def create_quiz(req: QuizRequest, request: Request):
    print("🚀 Received quiz request:", req)

    user_id = extract_user_id(request, req)
    print("🔍 Using user_id:", user_id)

    topic = req.topic.strip()
    difficulty = req.difficulty or "medium"
    count = max(1, min(req.question_count or 5, 10))
    types = req.question_types or ["multiple_choice"]

    context = get_cached_context(user_id)

    questions = generate_questions(
        topic=topic,
        difficulty=difficulty,
        count=count,
        types=types,
        context=context
    )

    quiz_id = f"quiz_{uuid.uuid4().hex[:6]}"
    summary = "; ".join(q.question for q in questions)

    log_learning_event(topic, summary, len(questions), user_id)

    return QuizResponse(quiz_id=quiz_id, questions=questions)
