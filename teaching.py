# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal
import openai
import os
import json

# Ensure your OpenAI key is loaded from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# --- Input schema: only description --- #
class TeachingRequest(BaseModel):
    description: str

# --- Output block schema --- #
class TeachingBlock(BaseModel):
    type: Literal["section", "bullet_list", "example", "diagram_hint", "interactive_tip"]
    title: str
    content: str

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- Prompt to GPT-3.5 --- #
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

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": req.description}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
        )

        raw_output = response["choices"][0]["message"]["content"]
        parsed_output = json.loads(raw_output)
        return parsed_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
