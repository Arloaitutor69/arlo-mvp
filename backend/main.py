
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os

from flashcard_generator import generate_flashcards
from feynman_feedback import get_feynman_feedback
from session_planner import generate_study_plan
from Chat_Quiz_Utils import generate_chat_response, generate_quiz

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class SessionRequest(BaseModel):
    subject: str
    duration_minutes: int
    notes_text: str

class FlashcardRequest(BaseModel):
    topic: str
    notes_text: str
    difficulty: str
    format: str

class FeynmanRequest(BaseModel):
    topic: str
    user_explanation: str

class ReviewSheetRequest(BaseModel):
    topic: str
    notes_text: str
    missed_flashcards: list[str]
    feynman_feedback: str

class ChatRequest(BaseModel):
    message: str

class QuizRequest(BaseModel):
    topic: str
    notes_text: str
    num_questions: int

# ---------- Endpoints ----------
@app.post("/session")
async def generate_session_plan(req: SessionRequest):
    try:
        plan = generate_study_plan(req.subject, req.duration_minutes, req.notes_text)
        return {"plan": plan}
    except Exception as e:
        return {"error": str(e)}

@app.post("/flashcards")
async def flashcards(req: FlashcardRequest):
    try:
        cards = generate_flashcards(req.topic, req.notes_text, req.difficulty, req.format)
        return {"flashcards": cards}
    except Exception as e:
        return {"error": str(e)}

@app.post("/feynman")
async def feynman(req: FeynmanRequest):
    try:
        feedback = get_feynman_feedback(req.topic, req.user_explanation)
        return {"feedback": feedback}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        reply = generate_chat_response(req.message)
        return {"response": reply}
    except Exception as e:
        return {"error": str(e)}

@app.post("/quiz")
async def quiz(req: QuizRequest):
    try:
        quiz_json = generate_quiz(req.topic, req.notes_text, req.num_questions)
        quiz = json.loads(quiz_json)
        return {"quiz": quiz}
    except Exception as e:
        return {"error": str(e)}
