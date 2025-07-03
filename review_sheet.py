# UPDATED REVIEW SHEET MODULE WITH SHARED IN-MODULE CONTEXT CACHE

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import openai
import os
import json
import requests
from datetime import datetime, timedelta
from threading import Thread

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

app = FastAPI()
router = APIRouter()

# ---------------------------
# In-memory Context Cache
# ---------------------------
context_cache = {}
context_ttl = timedelta(minutes=5)

def fetch_and_update_context(user_id: str):
    try:
        res = requests.get(f"{CONTEXT_API_BASE}/api/context/cache?user_id={user_id}", timeout=5)
        res.raise_for_status()
        context_cache[user_id] = (datetime.now(), res.json())
    except Exception as e:
        print("❌ Background context fetch failed:", e)

def get_cached_context(user_id: str):
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp > context_ttl:
            Thread(target=fetch_and_update_context, args=(user_id,)).start()
        return cached_value
    else:
        try:
            res = requests.get(f"{CONTEXT_API_BASE}/api/context/cache?user_id={user_id}", timeout=5)
            res.raise_for_status()
            context = res.json()
            context_cache[user_id] = (now, context)
            return context
        except Exception as e:
            print("❌ Initial context fetch failed:", e)
            return {}

# ---------------------------
# Pydantic Models
# ---------------------------
class ReviewRequest(BaseModel):
    user_id: Optional[str] = None

class ReviewSheet(BaseModel):
    summary: str
    memorization_facts: List[str]
    weak_areas: List[str]
    major_topics: List[str]

# ---------------------------
# Extract user_id from all valid locations
# ---------------------------
def extract_user_id(request: Request, data: ReviewRequest) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

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
        temperature=0.4,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ---------------------------
# Prompt Generator
# ---------------------------
def build_review_prompt(context: dict) -> str:
    return f"""
You are Arlo, a helpful AI tutor.
A student has just completed a study session. Based on the following structured context data, generate a bedtime review sheet.

Instructions:
- Focus primarily on the most recent learning_history entries.
- Use emphasized_facts, weak_areas, and user_goals if provided.
- Do not include topics that are not present in the context.

Context:
{json.dumps(context, indent=2)}

Respond ONLY in the following JSON format:
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
def generate_review_sheet(request: Request, data: ReviewRequest):
    user_id = extract_user_id(request, data)
    context = get_cached_context(user_id)
    prompt = build_review_prompt(context)

    print("📝 GPT prompt:\n", prompt)
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
        print("🟥 GPT RAW OUTPUT:\n", raw_output)
        print("❌ Error parsing GPT output as JSON:", e)
        raise HTTPException(status_code=500, detail="Review generation failed")

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001)
