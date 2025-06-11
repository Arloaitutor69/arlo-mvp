# ✅ Fully patched quiz module with valid learning_event logging

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
        res = requests.get(f"{CONTEXT_BASE_URL}/api/context/slice")
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print("❌ Context fetch failed:", e)
    return {}

def post_context_update(payload: dict):
    try:
        requests.post(f"{CONTEXT_BASE_URL}/api/context/update", json=payload)
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

        if content.startswith("```"):
            content = "\n".join(content.strip().splitlines()[1:-1])

        raw_questions = json.loads(content)
        return [QuizQuestion(**q) for q in raw_questions]

    except Exception as e:
        print("❌ GPT Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions from GPT")

# --- Route ---

@router.post("/api/quiz", response_model=QuizResponse)
async def create_quiz(data: QuizRequest):
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

    # Summarize questions to provide GPT trace for memory logging
    summary = "; ".join([q.question for q in questions])

    # ✅ Post structured learning_event to context manager
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
        }
    })

    return QuizResponse(quiz_id=quiz_id, questions=questions)
