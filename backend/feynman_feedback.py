# backend/feynman_feedback.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import openai
import json
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

class FeynmanRequest(BaseModel):
    concept: str
    user_explanation: str
    personalized_context: Optional[str] = None

class FeynmanResponse(BaseModel):
    message: str
    follow_up_question: Optional[str]
    action_suggestion: Optional[str] = "stay_in_phase"

@router.post("/api/feynman", response_model=FeynmanResponse)
async def run_feynman_phase(data: FeynmanRequest):
    try:
        prompt = f"""
You're ARLO, an excited AI tutor helping a student master topics using the Feynman technique.

Concept: {data.concept}
Student's Explanation: {data.user_explanation}
{f'Extra Context: {data.personalized_context}' if data.personalized_context else ''}

Instructions:
1. If correct but wordy or unclear, ask clarifying questions or say "Explain it like I'm 10."
2. If mostly right, fill in missing info and guide them.
3. If confused, explain from scratch.

Respond in this JSON format:
{{
  "message": "...",
  "follow_up_question": "...",
  "action_suggestion": "stay_in_phase"
}}
"""
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return json.loads(response.choices[0].message["content"])
    except Exception as e:
        return {
            "message": f"Oops! {str(e)}",
            "follow_up_question": "Can you explain that again?",
            "action_suggestion": "stay_in_phase"
        }

