from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import json
import os
from dotenv import load_dotenv

from flashcard_generator import generate_flashcards

from quiz import router as quiz_router  # ✅ import the new quiz module


# Load local .env variables
load_dotenv()

app = FastAPI()

# Allow CORS from your Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://405e367a-b787-41ce-904a-d1882e6a9b65.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(quiz_router)

# Root route
@app.get("/")
def root():
    return {"message": "ARLO backend is alive"}

# Health check route
@app.get("/ping")
def health_check():
    return {"status": "ok"}

# Optional: root for testing
@app.get("/")
def read_root():
    return {"message": "ARLO backend is running"}

# Flashcard input schema
class FlashcardRequest(BaseModel):
    topic: str
    content: str
    difficulty: str = "medium"
    count: int = 10

# Flashcard output schema
class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str

# Flashcard generation route
@app.post("/api/flashcards")
def create_flashcards(data: FlashcardRequest):
    raw = generate_flashcards(data.topic, data.content, data.difficulty)

    print("RAW GPT OUTPUT:\n", raw)  # 👈 log GPT response before parsing

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        print("JSON ERROR:", str(e))  # 👈 log parse failure
        return {
            "error": "AI returned invalid JSON",
            "raw": raw,
            "message": str(e)
        }

    flashcards = []
    for i, card in enumerate(parsed[:data.count]):
        flashcards.append(FlashcardItem(
            id=f"card_{i+1}",
            front=card.get("question", ""),
            back=card.get("answer", ""),
            difficulty=data.difficulty,
            category="basic_derivatives"
        ))

    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }
