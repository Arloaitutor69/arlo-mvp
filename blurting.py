# blurting.py

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
    context_prompt: Optional[str] = None  # New field

class BlurtingResponse(BaseModel):
    feedback: str
    missed_concepts: List[str]
    context_alignment: str  # New field

# ======================
# Prompt Generator
# ======================

def generate_blurting_prompt(topic: str, content_summary: Optional[str], blurted_response: str, context_prompt: Optional[str]) -> str:
    summary_block = f"\nSummary of key concepts:\n{content_summary}" if content_summary else ""
    context_block = f"\nAdditional context for evaluation:\n{context_prompt}" if context_prompt else ""
    
    return (
        f"You're an educational coach helping a student review their memory of the topic: \"{topic}\"."
        f"{summary_block}"
        f"{context_block}\n\n"
        f"The student wrote this from memory:\n\"\"\"\n{blurted_response}\n\"\"\"\n\n"
        "Evaluate their explanation. Return a JSON object with:\n"
        "- \"feedback\": a paragraph highlighting what they did well and gently pointing out what was missing.\n"
        "- \"missed_concepts\": a list of key ideas or facts they forgot or explained poorly.\n"
        "- \"context_alignment\": a short sentence describing how well their answer aligns with the context prompt or learning goal, if provided.\n\n"
        "Only return valid JSON."
    )

# ======================
# Main API Endpoint
# ======================

@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: BlurtingRequest):
    try:
        prompt = generate_blurting_prompt(
            request.topic,
            request.content_summary,
            request.blurted_response,
            request.context_prompt
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

