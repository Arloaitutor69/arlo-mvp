from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai
import os

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app and router
app = FastAPI()
router = APIRouter()

# ---------------------------
# Pydantic Models
# ---------------------------
class LogEntry(BaseModel):
    phase: str
    question: Optional[str] = None
    user_answer: Optional[str] = None
    correct: Optional[bool] = None
    gpt_feedback: Optional[str] = None
    timestamp: Optional[str] = None

class ReviewRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    weak_areas: List[str]
    phases_used: List[str]
    log: List[LogEntry]

class ReviewSheet(BaseModel):
    summary: str
    weak_areas: List[str]
    active_recall_questions: List[str]
    blurt_prompt: str
    mind_map_outline: List[str]

# ---------------------------
# Prompt Generator
# ---------------------------
def build_review_prompt(data: ReviewRequest) -> str:
    log_text = "\n".join([
        f"Phase: {entry.phase}\nQ: {entry.question}\nA: {entry.user_answer}\nCorrect: {entry.correct}\nFeedback: {entry.gpt_feedback}\n"
        for entry in data.log if entry.question
    ])

    prompt = f"""
You are Arlo, an AI tutor summarizing a student's session on the topic '{data.topic}'.
The session lasted {data.duration} minutes and used these techniques: {', '.join(data.phases_used)}.

The student struggled most with: {', '.join(data.weak_areas)}.
Here is what they worked on:

{log_text}

Create a bedtime review sheet. Include:
1. A warm session summary.
2. A short list of active recall questions.
3. A blurt prompt (ask them to recall from memory).
4. A basic mind map outline with key steps or subtopics.
"""
    return prompt

# ---------------------------
# GPT Call
# ---------------------------
def call_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Endpoint
# ---------------------------
@router.post("/api/review-sheet", response_model=ReviewSheet)
def generate_review_sheet(data: ReviewRequest):
    prompt = build_review_prompt(data)
    output = call_gpt(prompt)

    # Minimal parser (assumes formatted output, else fallback)
    lines = output.split("\n")
    summary = lines[0].strip()
    questions, mindmap = [], []
    blurt_prompt = "Write everything you remember about the topic."

    for line in lines:
        if line.startswith("-") and "?" in line:
            questions.append(line.strip("- "))
        elif "blurt" in line.lower():
            blurt_prompt = line.strip("- ")
        elif line.startswith("*") or (":" in line and "?" not in line):
            mindmap.append(line.strip("* "))

    return ReviewSheet(
        summary=summary,
        weak_areas=data.weak_areas,
        active_recall_questions=questions,
        blurt_prompt=blurt_prompt,
        mind_map_outline=mindmap
    )

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

# ---------------------------
# Local Test Server
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)
