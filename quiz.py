# quiz.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import openai
import os

# Load your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# --- Request and Response Models ---

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

# --- GPT-4 Question Generator ---

def call_gpt_for_quiz(topic: str, difficulty: str, count: int, types: List[str]) -> List[QuizQuestion]:
    system_msg = "You are an expert tutor generating quiz questions in JSON."
    user_msg = f"""
Generate {count} quiz questions about the topic: "{topic}", difficulty: "{difficulty}", using ONLY these types: {types}.
Format your response as a JSON list of objects with keys: id, type, question, options (null if not needed), correct_answer, and explanation.

Example output:
[
  {{
    "id": 1,
    "type": "multiple_choice",
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "B",
    "explanation": "..."
  }},
  ...
]
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or gpt-4 if enabled
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        # Parse JSON directly from response text
        import json
        json_data = response['choices'][0]['message']['content']
        raw_questions = json.loads(json_data)

        return [QuizQuestion(**q) for q in raw_questions]

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions from GPT")

# --- FastAPI Endpoint ---

@router.post("/api/quiz", response_model=QuizResponse)
async def create_quiz(data: QuizRequest):
    if not data.question_types:
        raise HTTPException(status_code=400, detail="Must include at least one question type.")

    quiz_id = f"quiz_{uuid.uuid4().hex[:6]}"
    questions = call_gpt_for_quiz(
        topic=data.topic,
        difficulty=data.difficulty,
        count=data.question_count,
        types=data.question_types
    )

    return QuizResponse(quiz_id=quiz_id, questions=questions)
