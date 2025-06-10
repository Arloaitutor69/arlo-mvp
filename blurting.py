# blurting.py (same level as flashcard_generator.py, quiz.py, etc.)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json

router = APIRouter()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# ======================
# Request and Response Models
# ======================

class BlurtingRequest(BaseModel):
    topic: str
    content_summary: Optional[str] = None
    blurted_response: str

class BlurtingResponse(BaseModel):
    feedback: str
    missed_concepts: List[str]

# ======================
# Prompt Generator
# ======================

def generate_blurting_prompt(topic: str, content_summary: Optional[str], blurted_response: str) -> str:
    context = f"\nContext:\n{content_summary}" if content_summary else ""
    return f"""
You're an educational coach helping a student review their memory of the topic: "{topic}".{context}

The student wrote this from memory:
"""{blurted_response}"""

Evaluate their explanation. Return a JSON object with:
- "feedback": a paragraph highlighting what they did well and gently pointing out what was missing.
- "missed_concepts": a list of key ideas or facts they forgot or explained poorly.

Only return valid JSON.
"""

# ======================
# Main API Endpoint
# ======================

@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: BlurtingRequest):
    try:
        prompt = generate_blurting_prompt(
            request.topic,
            request.content_summary,
            request.blurted_response
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        return BlurtingResponse(**parsed)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse GPT response as JSON.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
