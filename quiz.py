from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

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

# --- GPT Wrapper ---

def call_gpt_for_quiz(topic: str, difficulty: str, count: int, types: List[str]) -> List[QuizQuestion]:
    system_msg = "You are an expert tutor generating quiz questions in JSON."
    user_msg = f"""
Generate {count} quiz questions about the topic: "{topic}", difficulty: "{difficulty}", using ONLY these types: {types}.
Each question must include:
- id (integer)
- type (as in the list above)
- question (string)
- options (list of choices or null for true/false and fill-in-the-blank)
- correct_answer (string)
- explanation (string)

Return a valid JSON array. Do NOT wrap in markdown or include any notes or headings.
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

        # 🧼 Remove Markdown code wrapper if GPT adds one
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()

        raw_questions = json.loads(content)
        return [QuizQuestion(**q) for q in raw_questions]

    except Exception as e:
        print("❌ GPT Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate quiz questions from GPT")

# --- API Route ---

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
