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

class BlurtingRequest(BaseModel):
    topic: str
    user_blurting: str
    reference_notes: str

class MindmapRequest(BaseModel):
    topic: str
    notes_text: str

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

@app.post("/blurting-feedback")
def blurting_feedback(data: BlurtingRequest):
    prompt = f"""
    The student is practicing the blurting technique for the topic: {data.topic}.

    Below is their attempt to recall everything they know:
    """
    {data.user_blurting}
    """

    Compare it against the reference notes:
    """
    {data.reference_notes}
    """

    Identify any important concepts or details they missed or confused.
    Respond in bullet points.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"blurting_feedback": response.choices[0].message.content.strip()}

@app.post("/generate-mindmap")
def generate_mindmap(data: MindmapRequest):
    prompt = f"""
    For the topic "{data.topic}", generate a hierarchical mind map based on the following notes:
    {data.notes_text}

    Structure the response as a JSON dictionary:
    {
      "Central Idea": "...",
      "Branches": {
        "Branch 1": ["Subtopic A", "Subtopic B"],
        "Branch 2": ["Subtopic C"]
      }
    }
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"mindmap": response.choices[0].message.content.strip()}
from fastapi import Request
from pydantic import BaseModel

class TaskProgress(BaseModel):
    topic: str
    notes_text: str
    current_step: int
    user_input: str
    history: list

@app.post("/next-task")
def next_task(payload: TaskProgress):
    topic = payload.topic
    notes = payload.notes_text
    step = payload.current_step
    user_input = payload.user_input or ""
    history = payload.history or []

    system_prompt = (
        "You are ARLO, a highly intelligent and friendly AI study tutor. "
        "You're walking a student through an interactive study session on the topic: "
        f"'{topic}'. Their notes are:\n\n{notes}\n\n"
        "You must teach them the material clearly and conversationally, and actively use relevant study techniques (flashcards, Feynman, blurting, mind maps) to help them retain it. "
        "After each response from the student, you continue based on how they did. "
        "Be warm, focused, and step-by-step. Only generate one step at a time."
    )

    chat_history = [{"role": "system", "content": system_prompt}]

    # Append past turns
    for turn in history:
        chat_history.append({"role": "user", "content": turn["user"]})
        chat_history.append({"role": "assistant", "content": turn["arlo"]})

    # Append current turn
    if user_input:
        chat_history.append({"role": "user", "content": user_input})

    # Generate next step
    import openai
    import os
    openai.api_key = os.getenv("OPENAI_API_KEY")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or use gpt-4 if needed
            messages=chat_history,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        return {"arlo_reply": reply}
    except Exception as e:
        return {"arlo_reply": f"⚠️ ARLO had trouble generating the next step: {e}"}
