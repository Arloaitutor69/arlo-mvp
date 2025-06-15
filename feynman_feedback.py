from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import openai
import json
import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# Models
class FeynmanRequest(BaseModel):
    concept: str
    user_explanation: str
    personalized_context: Optional[str] = None
    user_id: Optional[str] = None

class FeynmanResponse(BaseModel):
    message: str
    follow_up_question: Optional[str]
    action_suggestion: Optional[str] = "stay_in_phase"

# Context helpers
def get_context_slice():
    try:
        res = requests.get(f"{CONTEXT_BASE}/api/context/slice", timeout=20)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return None

def post_learning_event_to_context(user_id: str, concept: str, feedback: str):
    user_id = user_id or "00000000-0000-0000-0000-000000000000"  # fallback for dev

    payload = {
        "source": f"user:{user_id}",
        "current_topic": concept,
        "weak_areas": [],
        "review_queue": [],
        "learning_event": {
            "concept": concept,
            "phase": "feynman",
            "confidence": 0.5,
            "depth": "intermediate",
            "source_summary": feedback[:250],
            "repetition_count": 1,
            "review_scheduled": True
        }
    }

    for attempt in range(3):
        try:
            res = requests.post(f"{CONTEXT_BASE}/api/context/update", json=payload, timeout=20)
            if res.status_code == 200:
                print("✅ Context logged successfully")
                return
            else:
                print(f"⚠️ Context log failed: {res.status_code}")
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            time.sleep(2)

# Endpoint
@router.post("/api/feynman", response_model=FeynmanResponse)
async def run_feynman_phase(data: FeynmanRequest):
    try:
        context_data = get_context_slice()
        if not data.personalized_context and context_data:
            data.personalized_context = json.dumps(context_data)

        prompt = f"""
You're ARLO, an excited AI tutor helping a student master topics using the Feynman technique.

Concept: {data.concept}
Student's Explanation: {data.user_explanation}
{f'Extra Context: {data.personalized_context}' if data.personalized_context else ''}

Instructions:
1. If correct but wordy or unclear, ask clarifying questions or say \"Explain it like I'm 10.\"
2. If mostly right, fill in missing info and guide them.
3. If confused, explain from scratch.

Respond in this JSON format:
{{
  "message": "...",
  "follow_up_question": "...",
  "action_suggestion": "stay_in_phase"
}}
"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        parsed = json.loads(response.choices[0].message["content"])

        if data.user_id:
            post_learning_event_to_context(data.user_id, data.concept, parsed["message"])

        return parsed

    except Exception as e:
        return {
            "message": f"Oops! {str(e)}",
            "follow_up_question": "Can you explain that again?",
            "action_suggestion": "stay_in_phase"
        }
