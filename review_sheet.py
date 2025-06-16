from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
import requests

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

app = FastAPI()
router = APIRouter()

# ---------------------------
# Pydantic Models
# ---------------------------
class ReviewSheet(BaseModel):
    summary: str
    memorization_facts: List[str]
    weak_areas: List[str]
    major_topics: List[str]

# ---------------------------
# Helper: Fetch Current Context
# ---------------------------
def fetch_context_slice():
    try:
        print("🧠 Fetching context slice...")
        res = requests.get(f"{CONTEXT_API_BASE}/api/context/slice", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to fetch context slice:", e)
        raise HTTPException(status_code=500, detail="Unable to fetch current context")

# ---------------------------
# GPT API Call
# ---------------------------
def call_gpt(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are Arlo, an expert learning coach generating review sheets."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Prompt Generator
# ---------------------------
def build_review_prompt(context: dict) -> str:
    return f"""
You are Arlo, a helpful AI tutor.
A student has just completed a study session. Based on the following structured context data, generate a bedtime review sheet.

Context:
{json.dumps(context, indent=2)}

Please include ONLY the following:
1. A short summary of what the student worked on and their general progress.
2. A list of key facts the student should memorize (especially those tied to errors or weak spots).
3. A list of topics that were covered during the session.
4. A list of weak areas the student should revisit.

Respond in pure JSON format as:
{{
  "summary": "...",
  "memorization_facts": ["..."],
  "major_topics": ["..."],
  "weak_areas": ["..."]
}}
"""

# ---------------------------
# Endpoint
# ---------------------------
@router.post("/review-sheet", response_model=ReviewSheet)
def generate_review_sheet():
    context = fetch_context_slice()
    prompt = build_review_prompt(context)
    raw_output = call_gpt(prompt)

    try:
        parsed = json.loads(raw_output)
        return ReviewSheet(
            summary=parsed.get("summary", ""),
            memorization_facts=parsed.get("memorization_facts", []),
            major_topics=parsed.get("major_topics", []),
            weak_areas=parsed.get("weak_areas", [])
        )
    except Exception as e:
        print("🔴 GPT RAW OUTPUT:\n", raw_output)
        print("❌ Error parsing GPT output as JSON:", e)
        raise HTTPException(status_code=500, detail="Review generation failed")

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)
