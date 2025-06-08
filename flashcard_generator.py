from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
import uuid

openai.api_key = os.getenv("OPENAI_API_KEY")
router = APIRouter()

# Input from planner or user
class FlashcardRequest(BaseModel):
    topic: str
    content: Optional[str] = ""
    difficulty: Optional[str] = "medium"
    count: Optional[int] = 10
    format: Optional[str] = "Q&A"

# Output flashcard schema
class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str

@router.post("/api/flashcards")
def generate_flashcards(data: FlashcardRequest):
    prompt = f"""
You are a flashcard tutor generating study cards from the following topic and notes.

Topic: "{data.topic}"
Notes: "{data.content or 'Use general knowledge if no notes provided.'}"

Difficulty: {data.difficulty}
Format: {data.format}

Use only one of these formats:
- "Q&A" → Basic question/answer
- "fill-in-the-blank" → Sentence with a missing term
- "multiple-choice" → (Not supported yet — just return Q&A for now)

Return ONLY a valid JSON array of objects like:
[
  {{ "question": "What is ...?", "answer": "..." }},
  {{ "question": "...?", "answer": "..." }}
]

Do not include explanations, headers, or any other text — just return the JSON array.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        raw = response.choices[0].message.content.strip()

        # Remove markdown block wrapper if GPT adds one
        if raw.startswith("```"):
            raw = "\n".join(raw.strip().splitlines()[1:-1])

        cards = json.loads(raw)

    except Exception as e:
        print("⚠️ Error:", e)
        raise HTTPException(status_code=500, detail="Failed to generate flashcards")

    flashcards = []
    for i, item in enumerate(cards[:data.count]):
        flashcards.append(FlashcardItem(
            id=f"card_{uuid.uuid4().hex[:6]}",
            front=item.get("question", "No question."),
            back=item.get("answer", "No answer."),
            difficulty=data.difficulty,
            category=data.topic
        ))

    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }
