from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import uuid
import os
import json
import openai
import requests

router = APIRouter()

# ✅ Add models
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

@router.get("/api/quiz/test")
def quiz_health_check():
    return {"status": "quiz models added, still good"}
