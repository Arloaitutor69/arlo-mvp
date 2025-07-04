# UPDATED FEYNMAN MODULE WITH SHARED IN-MODULE CONTEXT CACHE

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import openai
import os
import json
import requests
import logging
from datetime import datetime, timedelta
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
# In-memory Context Cache
# -----------------------------
context_cache = {}
context_ttl = timedelta(minutes=5)

def get_cached_context(user_id: str):
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return cached_value
    try:
        res = requests.get(f"{CONTEXT_BASE}/api/context/cache?user_id={user_id}", timeout=5)
        res.raise_for_status()
        context = res.json()
        context_cache[user_id] = (now, context)
        return context
    except Exception as e:
        logger.error("❌ Failed to fetch context: %s", str(e))
        return {}

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
# User ID extraction helper
# -----------------------------
def extract_user_id(request: Request, data: FeynmanRequest) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# -----------------------------
# Feynman endpoint
# -----------------------------
@router.post("/feynman", response_model=FeynmanResponse)
def run_feynman_phase(request: Request, payload: FeynmanRequest):
    logger.info("Received Feynman request for concept: %s", payload.concept)

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)

    # Build context summary string
    context_lines = []
    if context.get("current_topic"):
        context_lines.append(f"Current topic: {context['current_topic']}")
    if context.get("user_goals"):
        context_lines.append(f"Goals: {', '.join(context['user_goals'])}")
    if context.get("weak_areas"):
        context_lines.append(f"Weak areas: {', '.join(context['weak_areas'])}")
    if context.get("emphasized_facts"):
        context_lines.append(f"Focus points: {', '.join(context['emphasized_facts'])}")
    if context.get("preferred_learning_styles"):
        context_lines.append(f"Style: {context['preferred_learning_styles'][0]}")

    context_summary = "\n".join(context_lines)

    # Prompt construction
    prompt = (
        f"You are Arlo, an AI tutor helping a student master the concept of \"{payload.concept}\".\n"
        f"They just tried to explain it in their own words:\n\"{payload.user_explanation}\"\n\n"
        f"Use the personalized context and tutoring context below to guide your response.\n\n"
        f"{payload.personalized_context or context_summary}\n\n"
        "Your job:\n"
        "1. Gently point out any major gaps or mistakes in their explanation.\n"
        "2. Rephrase or clarify key points to improve understanding.\n"
        "3. Ask one helpful follow-up question to go deeper.\n"
        "Keep your tone warm and supportive."
    )

    logger.info("Constructed prompt:\n%s", prompt)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=500,
            request_timeout=15
        )

        raw_reply = response["choices"][0]["message"]["content"].strip()
        logger.info("OpenAI response: %s", raw_reply)

        # Try to extract a follow-up question (any line with a '?')
        lines = raw_reply.split("\n")
        follow_up = next((line for line in lines if "?" in line), None)

        return FeynmanResponse(
            message=raw_reply,
            follow_up_question=follow_up,
            action_suggestion="stay_in_phase"
        )

    except Exception as e:
        logger.exception("Feynman GPT call failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal AI error. Check logs.")
