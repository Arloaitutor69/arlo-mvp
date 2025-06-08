# quiz.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import random

# Set up router for quiz generation
router = APIRouter()

# Input model
class QuizRequest(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    question_count: int
    question_types: List[Literal["multiple_choice", "true_false"]]

# Output models
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

# Dummy quiz generation logic
def generate_questions(data: QuizRequest) -> List[QuizQuestion]:
    questions = []

    for i in range(1, data.question_count + 1):
        q_type = random.choice(data.question_types)

        if q_type == "multiple_choice":
            question = QuizQuestion(
                id=i,
                type="multiple_choice",
                question=f"What is a key part of {data.topic}?",
                options=["Nucleus", "Mitochondria", "Ribosome", "Vacuole"],
                correct_answer="Nucleus",
                explanation="The nucleus contains DNA and controls cell activities."
            )
        else:
            question = QuizQuestion(
                id=i,
                type="true_false",
                question=f"{data.topic} involves cellular activity.",
                correct_answer="True",
                explanation="Most biological processes in this topic involve cellular activity."
            )

        questions.append(question)

    return questions

# POST endpoint for quiz generation
@router.post("/api/quiz", response_model=QuizResponse)
async def create_quiz(data: QuizRequest):
    quiz_id = f"quiz_{uuid.uuid4().hex[:6]}"
    questions = generate_questions(data)
    return QuizResponse(quiz_id=quiz_id, questions=questions)
