# ✅ Debug-patched quiz module with context logging + test route

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import openai
import os
import json
import requests

openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

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

# --- Helpers ---

def fetch_context_slice():
    try:
        print("🔎 Fetching context from /context/slice...")
        res = requests.get(f"{CONTEXT_BASE_URL}/api/context/slice")
        res.raise_for_status()
        context = res.json()
        print("✅ Context received:", context)
        return context
    except Exception as e:
        print("❌ Context fetch failed:", e)
        return {}

def post_context_update(payload: dict):
    try:
        print("📤 Posting context update...")
        res = requests.post(f"{CONTEXT_BASE_URL}/api/context/update", json=payload)
        print("✅ Context update status:", res.status_code)
    except Exception as e:
        print("❌ Context update failed:", e)

# --- GPT Wrapper ---

def call_gpt_for_quiz(topic: str, difficulty: str, count: int, types: List[str], context: dict) -> List[QuizQuestion]:
    system_msg = "You are an expert tutor generating quiz questions in JSON format."

    current_topic = context.get("current_topic", "")
    weak_areas = ", ".join(context.get("weak_areas", []))
    emphasized_facts = ", ".join(context.get("emphasized_facts", []))
    user_goals = ", ".join(context.get("user_goals", []))
    review_queue = ", ".join(context.get("review_queue", []))

    user_msg = f"""
Generate {count} quiz questions about the topic: \"{topic}\" (context topic: \"{current_topic}\"), difficulty: \"{difficulty}\".

Allowed types: {types}

Please personalize:
- Prioritize these weak areas: {weak_areas or 'none'}
- Reinforce these emphasized facts: {emphasized_facts or 'none'}
- Tailor explanations to user goals: {user_goals or 'none'}
- Optionally include 1–2 review questions: {review_queue or 'none'}

Return ONLY a valid JSON array of objects like:
[
  {{
    "id": 1,
    "type": "multiple_choice",
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "...",
    "explanation": "..."
  }},
  ...
]
No extra text. No markdown.
"""

    try:
        print("⚙ Calling GPT for quiz generation...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.6,
            max_tokens=1200
        )

        content = response['choices'][0]['message']['content'].strip()
        print("🧠 GPT raw output:", content[:200])  # Preview output

        if content.startswith("```"):
            content = "\n".join(content.strip().splitlines()[1:-1])

        raw_questions = json.loads(content)
        return [QuizQuestion(**q) for q in raw_questions]

    except Exception as e:
        print("❌ GPT Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions from GPT")

# --- Routes ---

@router.post("/api/quiz", response_model=QuizResponse)
async def create_quiz(data: QuizRequest):
    print("🚀 /api/quiz called with:", data)

    if not data.question_types:
        raise HTTPException(status_code=400, detail="Must include at least one question type.")

    context = fetch_context_slice()

    quiz_id = f"quiz_{uuid.uuid4().hex[:6]}"
    questions = call_gpt_for_quiz(
        topic=data.topic,
        difficulty=data.difficulty,
        count=data.question_count,
        types=data.question_types,
        context=context
    )

    summary = "; ".join([q.question for q in questions])

    post_context_update({
        "source": "quiz",
        "phase": "quiz",
        "event_type": "generation",
        "learning_event": {
            "concept": data.topic,
            "phase": "quiz",
            "confidence": 0.5,
            "depth": "shallow",
            "source_summary": summary,
            "repetition_count": 1,
            "review_scheduled": False
        },
        "data": {
            "topic": data.topic,
            "difficulty": data.difficulty,
            "question_count": len(questions)
        }
    })

    return QuizResponse(quiz_id=quiz_id, questions=questions)

# --- Health Check ---

@router.get("/api/quiz/test")
def test_quiz_module():
    return {"status": "ok", "message": "Quiz module is live"}
