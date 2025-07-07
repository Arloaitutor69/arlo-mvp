# teaching.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Literal
import openai
import os
import json

router = APIRouter()

# --- Input schema --- #
class TeachingRequest(BaseModel):
    topic: str
    description: str
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]] = "beginner"
    learning_style: Optional[str] = "mixed"  # e.g. "visual", "auditory", etc.

# --- Output schema --- #
class TeachingBlock(BaseModel):
    type: Literal["section", "bullet_list", "example", "diagram_hint", "interactive_tip"]
    title: Optional[str]
    content: str

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- GPT Prompt Template --- #
GPT_SYSTEM_PROMPT = """
You are an expert teacher and memory coach. Your job is to teach the requested topic in a clear, structured, and digestible way, using proven learning techniques:
- Active recall
- Mnemonics
- Chunking
- Memory palace (descriptive visuals)
- Metaphors/examples (especially based on student interests)
- Bullet points + short paragraphs + definitions
- Humor and engagement when appropriate

Always:
- Start with a short summary paragraph
- Include labeled sections
- Use definitions + bullet points to explain terms

Keep language clear concise and informative.
Respond in JSON format using the following structure:
{{"lesson": [{{"type": "section", "title": "...", "content": "..."}}, ...]}}
"""

# --- Route --- #
@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        prompt = GPT_SYSTEM_PROMPT.format(learning_style=req.learning_style)

        user_prompt = f"""
Topic: {req.topic}
Goal: {req.description}
Difficulty: {req.difficulty}
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )

        raw = response["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
