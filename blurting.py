# backend/routers/blurting.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Request model
class BlurtingRequest(BaseModel):
    topic: str
    content_summary: Optional[str] = None
    blurted_response: str

# Response model
class BlurtingResponse(BaseModel):
    feedback: str
    missed_concepts: List[str]

# Prompt template
def generate_prompt(topic: str, summary: Optional[str], response: str) -> str:
    context = f"\nContext:\n{summary}" if summary else ""
    return f"""
You're an educational coach helping a student review their memory of "{topic}".{context}

They wrote:
\"\"\"{response}\"\"\"

Evaluate their explanation. Return only a valid JSON object with:
- "feedback": a paragraph highlighting what they did well and gently pointing out what was missing.
- "missed_concepts": a list of important facts or steps they forgot.

Only return JSON. Be clear and supportive.
"""

@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: BlurtingRequest):
    try:
        prompt = generate_prompt(
            request.topic,
            request.content_summary,
            request.blurted_response
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return BlurtingResponse(**parsed)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not parse GPT response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
