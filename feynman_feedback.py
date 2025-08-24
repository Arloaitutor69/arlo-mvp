from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Literal
import os
import json
import requests
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feynman_teaching")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Context API configuration
CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Context Cache
# -----------------------------
context_cache = {}
context_ttl = timedelta(minutes=10)

def get_cached_context(user_id: str):
    """Optimized context retrieval with intelligent caching"""
    now = datetime.now()
    cache_key = f"ctx_{user_id}"
    
    if cache_key in context_cache:
        timestamp, cached_value = context_cache[cache_key]
        if now - timestamp < context_ttl:
            return cached_value
    
    try:
        res = requests.get(f"{CONTEXT_BASE}/api/context/cache?user_id={user_id}", timeout=3)
        res.raise_for_status()
        context = res.json()
        context_cache[cache_key] = (now, context)
        return context
    except Exception as e:
        logger.error("❌ Context fetch failed: %s", str(e))
        return {}

# -----------------------------
# Pydantic Models
# -----------------------------
class FeynmanExerciseRequest(BaseModel):
    teaching_content: str
    user_id: Optional[str] = None
    difficulty_level: Optional[str] = "intermediate"
    subject_area: Optional[str] = None

class FeynmanExerciseResponse(BaseModel):
    questions: List[str]

class FeynmanAssessmentRequest(BaseModel):
    question: str
    user_explanation: str
    user_id: Optional[str] = None

class FeynmanAssessmentResponse(BaseModel):
    mastery_score: int
    what_went_well: List[str]
    gaps_in_understanding: List[str]

# -----------------------------
# JSON Schemas for structured outputs
# -----------------------------
FEYNMAN_EXERCISE_SCHEMA = {
    "name": "feynman_exercise_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "string",
                    "minLength": 20
                }
            }
        },
        "required": ["questions"],
        "additionalProperties": False
    }
}

FEYNMAN_ASSESSMENT_SCHEMA = {
    "name": "feynman_assessment_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "mastery_score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100
            },
            "what_went_well": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 10
                }
            },
            "gaps_in_understanding": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "string",
                    "minLength": 15
                }
            }
        },
        "required": ["mastery_score", "what_went_well", "gaps_in_understanding"],
        "additionalProperties": False
    }
}

# -----------------------------
# User ID extraction helper
# -----------------------------
def extract_user_id(request: Request, data) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif hasattr(data, 'user_id') and data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# -----------------------------
# Enhanced Feynman Exercise Generator
# -----------------------------
@router.post("/feynman/exercises", response_model=FeynmanExerciseResponse)
def generate_feynman_exercises(request: Request, payload: FeynmanExerciseRequest):
    logger.info("🎯 Generating Feynman exercises")

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    system_prompt = """You are an expert AI tutor creating Feynman-style teaching exercises for conceptual mastery.

Create exactly 3 conceptual teaching questions that:
- Test deep understanding, not memorization
- Are relevant to the teaching material and what a student might see on a test
- Help students explain concepts in their own words

EXAMPLE FORMAT:
Content: The Age of Exploration was a period between the 15th and 17th centuries when European powers expanded across the globe through sea voyages, driven by economic motives, technological advances, and cultural imperatives like spreading Christianity.

Expected Questions:
1. Why did European exploration expand so rapidly in the late 15th century, and what made this timing significant compared to earlier periods?
2. How did the principles of mercantilism influence the goals and outcomes of European exploration and colonization?  
3. How did the Columbian Exchange fundamentally transform both European and non-European societies — economically, culturally, and biologically?

Return exactly 3 questions that encourage deep conceptual understanding."""

    user_prompt = f"""TEACHING CONTENT:
"{payload.teaching_content}"

Create exactly 3 conceptual teaching questions relevant to this material."""

    try:
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": FEYNMAN_EXERCISE_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        logger.info("✅ Exercise generation completed")
        
        return FeynmanExerciseResponse(
            questions=parsed_data["questions"]
        )

    except json.JSONDecodeError as e:
        # This should never happen with structured outputs, but just in case
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse response as JSON: {str(e)}"
        )
    
    except Exception as e:
        logger.exception("❌ Exercise generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Exercise generation service temporarily unavailable")

# -----------------------------
# Enhanced Feynman Assessment
# -----------------------------
@router.post("/feynman/assess", response_model=FeynmanAssessmentResponse)
def assess_feynman_teaching(request: Request, payload: FeynmanAssessmentRequest):
    logger.info("🎓 Assessing Feynman teaching")

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    system_prompt = """You are an expert AI tutor assessing a student's conceptual explanation using the Feynman Technique.

Assess the student's explanation and provide:
1. A mastery score out of 100 (be precise and fair)
2. What they did well (specific strengths)
3. Gaps in understanding (specific areas needing improvement)

Focus on:
- Conceptual accuracy and depth
- Use of appropriate terminology
- Recognition of complexity and nuance
- Ability to connect ideas

EXAMPLE ASSESSMENT:
Question: Why did European exploration expand so rapidly in the late 15th century?
Student Answer: "Exploration happened because countries wanted land and they found new places and took them. They had ships and went far and people got rich."

Expected Assessment:
Mastery Score: 48
What You Did Well:
- Recognized that European nations were seeking new land and wealth
- Noted that ships were a key factor in enabling exploration

Gaps in Understanding:
- No mention of historical context — why the 15th century was a turning point (e.g., Fall of Constantinople, Renaissance thought, Ottoman control of land routes)
- Oversimplified motivations — didn't explain religious motives, competition among European powers, or desire for direct access to Asian trade
- Vague use of terms like "people got rich" without referring to the Columbian Exchange, mercantilism, or monarchial sponsorship
- No recognition of the impact on indigenous societies or the idea of cultural imperialism"""

    user_prompt = f"""QUESTION: "{payload.question}"

STUDENT'S EXPLANATION:
"{payload.user_explanation}"

Assess this student explanation and provide a detailed evaluation."""

    try:
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": FEYNMAN_ASSESSMENT_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        logger.info("✅ Assessment completed")

        return FeynmanAssessmentResponse(
            mastery_score=parsed_data["mastery_score"],
            what_went_well=parsed_data["what_went_well"],
            gaps_in_understanding=parsed_data["gaps_in_understanding"]
        )

    except json.JSONDecodeError as e:
        # This should never happen with structured outputs, but just in case
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse response as JSON: {str(e)}"
        )
    
    except Exception as e:
        logger.exception("❌ Assessment failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Assessment service temporarily unavailable")
