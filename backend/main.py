from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env (for local dev)
load_dotenv()

# App instance
app = FastAPI()

# Allow cross-origin requests from Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Lovable domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import List
import json
from flashcard_generator import generate_flashcards

# Input format from Lovable
class FlashcardRequest(BaseModel):
    topic: str
    content: str
    difficulty: str = "medium"
    count: int = 10

# Output card schema
class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str

@app.post("/api/flashcards")
def create_flashcards(data: FlashcardRequest):
    raw = generate_flashcards(data.topic, data.content, data.difficulty)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "AI returned invalid JSON", "raw": raw}

    flashcards = []
    for i, card in enumerate(parsed[:data.count]):
        flashcards.append(FlashcardItem(
            id=f"card_{i+1}",
            front=card.get("question", ""),
            back=card.get("answer", ""),
            difficulty=data.difficulty,
            category="basic_derivatives"  # Can update logic here later
        ))

    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }

# Example Supabase call
@app.get("/supabase-users")
def get_users():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    response = requests.get(
        f"{supabase_url}/rest/v1/users",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        },
    )
    return response.json()

