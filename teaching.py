# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union
import openai
import os
import json

# Load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# --- Input schema: only description --- #
class TeachingRequest(BaseModel):
    description: str

# --- Output schema --- #
class TeachingBlock(BaseModel):
    type: Literal["section", "bullet_list", "example", "diagram_hint", "interactive_tip"]
    title: str
    content: Union[str, List[str]]  # support both plain strings and structured lists

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- GPT Prompt --- #
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
- Include clearly labeled sections
- Use bullet points and short paragraphs to explain terms
- Define key vocabulary in plain language
- Include helpful metaphor or analogy to deepen understanding, visual suggestion (diagram or drawing idea), active recall question or exercise, and one mnemonic or memory strategy
- If listing bullet points, use type: "bullet_list" and make content an array of strings

Ensure your output is detailed and thorough, providing enough content to fully teach the content. Prioritize clarity and usefulness over brevity. Make sure to be comprehensive and return a lot of output.

Respond in JSON format using the following structure:
{"lesson": [{"type": "section", "title": "...", "content": "..."}, ...]}
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

        # Cleanup: ensure valid structure
        for i, block in enumerate(parsed_output.get("lesson", [])):

            # Fallback: insert title if missing
            if not block.get("title"):
                block["title"] = f"Part {i + 1}"

            # Normalize bullet lists
            content = block.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, dict) and "content" in item:
                        new_content.append(item["content"])
                    elif isinstance(item, str):
                        new_content.append(item)
                block["content"] = new_content

        return parsed_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
