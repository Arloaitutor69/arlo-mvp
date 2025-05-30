
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import json

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

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

@app.post("/flashcards")
async def generate_flashcards(req: FlashcardRequest):
    prompt = f"Generate 5 {req.difficulty} {req.format} flashcards for this topic: {req.topic}.\nNotes: {req.notes_text}\nFormat: JSON with 'question' and 'answer' keys."
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        cards = json.loads(result["choices"][0]["message"]["content"])
        return {"flashcards": cards}
    except Exception as e:
        return {"error": str(e)}

@app.post("/feynman")
async def feynman_feedback(req: FeynmanRequest):
    prompt = f"Evaluate this explanation using the Feynman technique and provide clear feedback:\nTopic: {req.topic}\nExplanation: {req.user_explanation}"
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"feedback": result["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat_response(req: ChatRequest):
    prompt = f"You are an AI tutor. Respond helpfully to: {req.message}"
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return {"response": result["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/quiz")
async def generate_quiz(req: QuizRequest):
    prompt = f"Create a {req.num_questions}-question multiple-choice quiz on '{req.topic}' using the notes below:\n{req.notes_text}\nReturn JSON: list of questions with 'question', 'choices' (list), and 'answer'."
    try:
        result = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        quiz = json.loads(result["choices"][0]["message"]["content"])
        return {"quiz": quiz}
    except Exception as e:
        return {"error": str(e)}
