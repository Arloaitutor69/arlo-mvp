from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import openai
import os
import json
import requests
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import hashlib
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("feynman_teaching")

# OpenAI and Context API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
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
    logger.info("🎯 Generating Feynman exercises for: %s", payload.topic)

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    prompt = f"""You are an expert AI tutor creating Feynman-style teaching exercises for conceptual mastery.

TEACHING CONTENT:
"{payload.teaching_content}"

YOUR TASK:
Create exactly 3 conceptual teaching questions relevant to teaching material and what a student might see on a test:
- Test deep understanding, not memorization
- Use the format: *Question text ending with question mark?*
- align your output formatting and the qaulity with style of example below 

EXAMPLE FORMAT 
Content: The Age of Exploration was a period between the 15th and 17th centuries when European powers expanded across the globe through sea voyages, driven by economic motives, technological advances, and cultural imperatives like spreading Christianity.

Expected Output:
1. *Why did European exploration expand so rapidly in the late 15th century, and what made this timing significant compared to earlier periods?*
2. *How did the principles of mercantilism influence the goals and outcomes of European exploration and colonization?*
3. *How did the Columbian Exchange fundamentally transform both European and non-European societies — economically, culturally, and biologically?*

Respond with ONLY the 3 questions in the exact format shown above, numbered 1-3."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor specializing in creating Feynman-style conceptual teaching questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400,
            request_timeout=10
        )

        content = response["choices"][0]["message"]["content"].strip()
        logger.info("✅ Exercise generation completed")

        # Extract questions from response
        questions = []
        lines = content.split('\n')
        for line in lines:
            # Look for numbered questions with asterisks
            if re.match(r'^\d+\.\s*\*.*\*\s*$', line.strip()):
                question_text = re.sub(r'^\d+\.\s*\*(.*)\*\s*$', r'\1', line.strip())
                questions.append(question_text)

        if len(questions) != 3:
            # Fallback parsing if format doesn't match exactly
            questions = []
            for line in lines:
                if '?' in line and len(line.strip()) > 10:
                    clean_question = re.sub(r'^\d+\.\s*[\*]*\s*', '', line.strip())
                    clean_question = clean_question.replace('*', '').strip()
                    if clean_question:
                        questions.append(clean_question)

        # Ensure we have exactly 3 questions
        questions = questions[:3]

        return FeynmanExerciseResponse(
            topic=payload.topic,
            questions=questions
        )

    except Exception as e:
        logger.exception("❌ Exercise generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Exercise generation service temporarily unavailable")

# -----------------------------
# Enhanced Feynman Assessment
# -----------------------------
@router.post("/feynman/assess", response_model=FeynmanAssessmentResponse)
def assess_feynman_teaching(request: Request, payload: FeynmanAssessmentRequest):
    logger.info("🎓 Assessing Feynman teaching for: %s", payload.topic)

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    prompt = f"""You are an expert AI tutor assessing a student's conceptual explanation using the Feynman Technique.
    
QUESTION: "{payload.question}"

STUDENT'S EXPLANATION:
"{payload.user_explanation}"

EXAMPLE ASSESSMENT (based on Age of Exploration):
Question: Why did European exploration expand so rapidly in the late 15th century?
Student Answer: "Exploration happened because countries wanted land and they found new places and took them. They had ships and went far and people got rich."

Expected Assessment:
Mastery Score: 48/100

What You Did Well:
* ✅ Recognized that European nations were seeking new land and wealth
* ✅ Noted that ships were a key factor in enabling exploration

Gaps in Understanding:
* ❌ No mention of historical context — why the 15th century was a turning point (e.g., Fall of Constantinople, Renaissance thought, Ottoman control of land routes)
* ❌ Oversimplified motivations — didn't explain religious motives, competition among European powers, or desire for direct access to Asian trade
* ❌ Vague use of terms like "people got rich" without referring to the Columbian Exchange, mercantilism, or monarchial sponsorship
* ❌ No recognition of the impact on indigenous societies or the idea of cultural imperialism

YOUR TASK:
Assess the student's explanation and provide:

1. A mastery score out of 100 (be precise and fair)
2. What they did well (bullet points using * ✅ format)
3. Gaps in understanding (bullet points using * ❌ format)

Focus on:
- Conceptual accuracy and depth
- Use of appropriate terminology
- Recognition of complexity and nuance
- Ability to connect ideas

Format your response EXACTLY as:
Mastery Score: [X]/100

What You Did Well:
* ✅ [specific strength]
* ✅ [specific strength]

Gaps in Understanding:
* ❌ [specific gap with explanation]
* ❌ [specific gap with explanation]
* ❌ [specific gap with explanation]"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor specializing in conceptual mastery assessment using the Feynman Technique."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for consistent assessment
            max_tokens=600,
            request_timeout=10
        )

        content = response["choices"][0]["message"]["content"].strip()
        logger.info("✅ Assessment completed")

        # Parse the response
        mastery_score = 0
        what_went_well = []
        gaps = []

        # Extract mastery score
        score_match = re.search(r'Mastery Score:\s*(\d+)(?:/100)?', content)
        if score_match:
            mastery_score = int(score_match.group(1))

        # Extract what went well
        well_section = re.search(r'What You Did Well:(.*?)(?=Gaps in Understanding:|$)', content, re.DOTALL)
        if well_section:
            well_lines = well_section.group(1).split('\n')
            for line in well_lines:
                if '✅' in line:
                    clean_line = re.sub(r'^\s*\*\s*✅\s*', '', line.strip())
                    if clean_line:
                        what_went_well.append(clean_line)

        # Extract gaps
        gaps_section = re.search(r'Gaps in Understanding:(.*?)$', content, re.DOTALL)
        if gaps_section:
            gap_lines = gaps_section.group(1).split('\n')
            for line in gap_lines:
                if '❌' in line:
                    clean_line = re.sub(r'^\s*\*\s*❌\s*', '', line.strip())
                    if clean_line:
                        gaps.append(clean_line)

        return FeynmanAssessmentResponse(
            mastery_score=mastery_score,
            what_went_well=what_went_well,
            gaps_in_understanding=gaps
        )

    except Exception as e:
        logger.exception("❌ Assessment failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Assessment service temporarily unavailable")

