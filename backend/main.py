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

@app.post("/generate-session")
def generate_session(data: SessionRequest):
    prompt = f"""
    You are a study coach. Create a JSON object with a structured study plan for the topic '{data.subject}' using the user's notes below.
    Include these fields:
    - pomodoro: recommended Pomodoro structure (e.g., "25/5")
    - techniques: a list of the study methods to be used
    - tasks: step-by-step tasks using those methods
    - review_sheet: final bedtime summary

    NOTES:
    {data.notes_text}
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"session_plan": response.choices[0].message.content.strip()}

@app.post("/generate-flashcards")
def generate_flashcards(data: FlashcardRequest):
    prompt = f"""
    Create 10 {data.difficulty} flashcards in {data.format} format for the topic: {data.topic}.
    Only use the following context:
    {data.notes_text}
    Format your response as a JSON list.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"flashcards": response.choices[0].message.content.strip()}

@app.post("/feynman-feedback")
def feynman_feedback(data: FeynmanRequest):
    prompt = f"""
    Evaluate this student's explanation of '{data.topic}':
    """
    {data.user_explanation}
    """
    Provide 2 parts:
    - feedback: how clear and correct it is
    - follow_up_questions: 2 questions to test their understanding

    Respond in JSON with keys: feedback and follow_up_questions (as a list).
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"feynman_response": response.choices[0].message.content.strip()}

@app.post("/generate-review-sheet")
def generate_review_sheet(data: ReviewSheetRequest):
    prompt = f"""
    Based on the topic "{data.topic}", the student's notes, missed flashcards, and Feynman explanation feedback,
    generate a review sheet with the following:

    - Key facts they struggled with
    - Any small but important details they should memorize
    - A brief final summary to review before bed

    NOTES:
    {data.notes_text}

    MISSED FLASHCARDS:
    {json.dumps(data.missed_flashcards)}

    FEEDBACK:
    {data.feynman_feedback}

    Format the output as bullet points.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"review_sheet": response.choices[0].message.content.strip()}
