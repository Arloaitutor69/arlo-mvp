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
    memorization_facts: List[str]
    explanations: List[str]

# ---------------------------
# Prompt Generator
# ---------------------------
def build_review_prompt(data: ReviewRequest) -> str:
    log_text = "\n".join([
        f"Q: {entry.question}\nA: {entry.user_answer}\nCorrect: {entry.correct}\nFeedback: {entry.gpt_feedback}\n"
        for entry in data.log if entry.question
    ])

    prompt = f"""
You are Arlo, a helpful AI tutor.
The student just finished a study session on "{data.topic}" using: {', '.join(data.phases_used)}.
It lasted {data.duration} minutes.

Their weak areas were: {', '.join(data.weak_areas)}.
Here is the session log:
{log_text}

Based on this, generate a bedtime review sheet with ONLY:
1. A short personalized summary of what the student worked on and struggled with.
2. A list of key facts the student should memorize before bed (especially from missed questions).
3. Simple, clear explanations of any concepts they misunderstood.

Output in JSON:
{{
  "summary": "...",
  "memorization_facts": ["..."],
  "explanations": ["..."]
}}
"""
    return prompt

# ---------------------------
# Endpoint
# ---------------------------
@router.post("/api/review-sheet", response_model=ReviewSheet)
def generate_review_sheet(data: ReviewRequest):
    prompt = build_review_prompt(data)
    raw_output = call_gpt(prompt)

    try:
        parsed = eval(raw_output)
        return ReviewSheet(
            summary=parsed.get("summary", ""),
            memorization_facts=parsed.get("memorization_facts", []),
            explanations=parsed.get("explanations", [])
        )
    except Exception:
        return ReviewSheet(
            summary="Unable to parse GPT response.",
            memorization_facts=[],
            explanations=[]
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
