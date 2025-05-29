from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from session_planner import generate_study_plan
from flashcard_generator import generate_flashcards
from feynman_feedback import get_feynman_feedback


app = FastAPI()

# Define the structure of the POST request data
class SessionRequest(BaseModel):
    subject: str
    duration_minutes: int
    notes_text: str = None

@app.get("/")
def read_root():
    return {"message": "Hello from ARLO backend!"}

@app.post("/generate-session")
def generate_session(request: SessionRequest):
    try:
        plan = generate_study_plan(
            subject=request.subject,
            duration_minutes=request.duration_minutes,
            notes_text=request.notes_text
        )
        return {"session_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FlashcardRequest(BaseModel):
    topic: str
    notes_text: str
    difficulty: str = "medium"
    format: str = "Q&A"

@app.post("/generate-flashcards")
def flashcards(request: FlashcardRequest):
    try:
        cards = generate_flashcards(
            topic=request.topic,
            notes_text=request.notes_text,
            difficulty=request.difficulty,
            format=request.format
        )
        return {"flashcards": cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class FeynmanRequest(BaseModel):
    topic: str
    user_explanation: str

@app.post("/feynman-feedback")
def feynman_feedback(request: FeynmanRequest):
    try:
        result = get_feynman_feedback(
            topic=request.topic,
            user_explanation=request.user_explanation
        )
        return {"feynman_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


