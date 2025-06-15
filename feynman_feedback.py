from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import openai
import os
import json
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI and Context API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Pydantic models
# -----------------------------
class FeynmanRequest(BaseModel):
    concept: str
    user_explanation: str
    personalized_context: Optional[str] = None
    user_id: Optional[str] = None

class FeynmanResponse(BaseModel):
    message: str
    follow_up_question: Optional[str]
    action_suggestion: Optional[str] = "stay_in_phase"

# -----------------------------
# Helper to get context
# -----------------------------
def get_context_slice():
    try:
        logger.info("Fetching context slice from: %s", CONTEXT_BASE)
        res = requests.get(f"{CONTEXT_BASE}/api/context/slice", timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error("Failed to fetch context slice: %s", str(e))
        return {}

# -----------------------------
# Feynman endpoint
# -----------------------------
@router.post("/feynman", response_model=FeynmanResponse)
def run_feynman_phase(payload: FeynmanRequest):
    logger.info("Received Feynman request for concept: %s", payload.concept)

    context = get_context_slice()
    prompt = (
        f"You are an AI tutor helping a student master the concept of {payload.concept}.\n"
        f"They just explained it like this:\n"
        f"\"\"\"\n{payload.user_explanation}\n\"\"\"\n"
        f"Your task is to provide kind, constructive feedback. Use their context to guide you:\n"
        f"Context: {payload.personalized_context or context}\n"
        "1. Identify any major gaps or inaccuracies in their explanation.\n"
        "2. Provide a revised explanation or clarification if needed.\n"
        "3. Ask a follow-up question to probe deeper understanding."
    )

    logger.info("Constructed prompt:\n%s", prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500
        )

        raw_reply = response["choices"][0]["message"]["content"]
        logger.info("OpenAI response: %s", raw_reply)

        # Simple structured parsing
        split_parts = raw_reply.split("\n")
        message = "\n".join(split_parts[:3]).strip()
        follow_up = next((line for line in split_parts if "?" in line), None)

        return FeynmanResponse(
            message=message,
            follow_up_question=follow_up,
            action_suggestion="stay_in_phase"
        )

    except Exception as e:
        logger.exception("Feynman GPT call failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal AI error. Check logs.")
