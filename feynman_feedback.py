# ENHANCED FEYNMAN MODULE - OPTIMIZED FOR TEACHING EXCELLENCE

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
logger = logging.getLogger("feynman_feedback")

# OpenAI and Context API configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Enhanced Context Cache with Compression
# -----------------------------
context_cache = {}
context_ttl = timedelta(minutes=10)  # Extended for better performance

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
# Enhanced Pydantic Models
# -----------------------------
class FeynmanRequest(BaseModel):
    concept: str
    user_explanation: str
    personalized_context: Optional[str] = None
    user_id: Optional[str] = None
    difficulty_level: Optional[str] = "intermediate"  # beginner, intermediate, advanced
    subject_area: Optional[str] = None  # biology, chemistry, physics, etc.

class FeynmanResponse(BaseModel):
    message: str
    follow_up_question: Optional[str]
    action_suggestion: Optional[str] = "stay_in_phase"
    concept_mastery_score: Optional[int] = None  # 1-100 scale
    key_gaps: Optional[List[str]] = None
    strengths: Optional[List[str]] = None

class FeynmanExerciseRequest(BaseModel):
    concept: str
    teaching_block: str
    user_id: Optional[str] = None
    difficulty_level: Optional[str] = "intermediate"
    subject_area: Optional[str] = None
    focus_areas: Optional[List[str]] = None  # specific areas to emphasize

class FeynmanExerciseResponse(BaseModel):
    exercise_1: dict
    exercise_2: dict
    bonus_challenge: Optional[dict] = None

# -----------------------------
# Teaching Enhancement Functions
# -----------------------------
def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from teaching content"""
    # Simple keyword extraction - could be enhanced with NLP
    concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    return list(set(concepts))[:5]  # Top 5 concepts

def generate_context_hash(user_id: str, concept: str) -> str:
    """Generate hash for context caching"""
    return hashlib.md5(f"{user_id}_{concept}".encode()).hexdigest()[:8]

def build_enhanced_context(context: dict, concept: str, subject_area: str = None) -> str:
    """Build rich context for better AI responses"""
    context_parts = []
    
    # Core learning profile
    if context.get("current_topic"):
        context_parts.append(f"📚 Current Topic: {context['current_topic']}")
    if context.get("user_goals"):
        context_parts.append(f"🎯 Goals: {', '.join(context['user_goals'][:3])}")
    if context.get("weak_areas"):
        context_parts.append(f"⚠️ Challenge Areas: {', '.join(context['weak_areas'][:3])}")
    if context.get("preferred_learning_styles"):
        style = context['preferred_learning_styles'][0] if context['preferred_learning_styles'] else "visual"
        context_parts.append(f"🧠 Learning Style: {style}")
    
    # Subject-specific context
    if subject_area:
        context_parts.append(f"📖 Subject: {subject_area}")
    
    return "\n".join(context_parts)

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
# Enhanced Feynman Feedback Endpoint
# -----------------------------
@router.post("/feynman", response_model=FeynmanResponse)
def run_feynman_phase(request: Request, payload: FeynmanRequest):
    logger.info("🎓 Feynman feedback for: %s", payload.concept)

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    # Build enhanced context
    enhanced_context = build_enhanced_context(context, payload.concept, payload.subject_area)
    
    # Construct optimized prompt for better teaching
    prompt = f"""You are Arlo, an expert AI tutor specializing in deep conceptual understanding. A student is learning "{payload.concept}" and just attempted to explain it.

STUDENT'S EXPLANATION:
"{payload.user_explanation}"

LEARNING CONTEXT:
{payload.personalized_context or enhanced_context}

DIFFICULTY LEVEL: {payload.difficulty_level}

YOUR TEACHING MISSION:
1. **Immediate Affirmation** (1 sentence): Highlight what they understood correctly
2. **Gap Analysis** (2-3 bullet points): Identify specific misconceptions or missing pieces
3. **Conceptual Clarification** (2-3 sentences): Explain the gaps using analogies or simpler terms
4. **Mastery Check** (1 question): Ask a follow-up that tests deeper understanding

TEACHING PRINCIPLES:
- Be encouraging but precise
- Use analogies relevant to their learning style
- Focus on "why" and "how" rather than just "what"
- Connect to real-world applications
- Identify test-relevant gaps

Provide a mastery score (1-100) based on their explanation completeness and accuracy."""

    logger.info("📝 Constructed enhanced prompt")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor focused on deep conceptual understanding and test preparation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,  # Slightly lower for more consistent feedback
            max_tokens=600,   # Increased for richer feedback
            request_timeout=10
        )

        raw_reply = response["choices"][0]["message"]["content"].strip()
        logger.info("✅ OpenAI response received")

        # Parse the response for structured feedback
        lines = raw_reply.split("\n")
        follow_up = next((line.strip() for line in lines if "?" in line and len(line.strip()) > 10), None)
        
        # Extract mastery score
        score_match = re.search(r'(\d+)(?:/100|%)', raw_reply)
        mastery_score = int(score_match.group(1)) if score_match else None
        
        # Extract key gaps and strengths
        gaps = []
        strengths = []
        
        for line in lines:
            if any(word in line.lower() for word in ['gap', 'missing', 'unclear', 'incorrect']):
                gaps.append(line.strip())
            elif any(word in line.lower() for word in ['correct', 'good', 'well', 'accurate']):
                strengths.append(line.strip())

        return FeynmanResponse(
            message=raw_reply,
            follow_up_question=follow_up,
            action_suggestion="stay_in_phase",
            concept_mastery_score=mastery_score,
            key_gaps=gaps[:3],  # Top 3 gaps
            strengths=strengths[:2]  # Top 2 strengths
        )

    except Exception as e:
        logger.exception("❌ Feynman GPT call failed: %s", str(e))
        raise HTTPException(status_code=500, detail="AI tutoring service temporarily unavailable")

# -----------------------------
# Enhanced Feynman Exercise Generator
# -----------------------------
@router.post("/feynman/exercises", response_model=FeynmanExerciseResponse)
def generate_feynman_exercises(request: Request, payload: FeynmanExerciseRequest):
    logger.info("🎯 Generating test-relevant exercises for: %s", payload.concept)

    user_id = extract_user_id(request, payload)
    context = get_cached_context(user_id)
    
    # Extract key concepts for targeted exercises
    key_concepts = extract_key_concepts(payload.teaching_block)
    
    prompt = f"""You are an expert AI tutor creating Feynman-style exercises for deep conceptual understanding.

CONCEPT: "{payload.concept}"
DIFFICULTY: {payload.difficulty_level}
SUBJECT: {payload.subject_area or "general"}

TEACHING CONTENT:
"{payload.teaching_block}"

KEY CONCEPTS IDENTIFIED: {', '.join(key_concepts)}

EXERCISE DESIGN CRITERIA:
- Target common test question patterns
- Require explanation, not just memorization
- Include real-world applications
- Test causal relationships and mechanisms
- Encourage teaching to others

Create 2 exercises + 1 bonus challenge that would help students ace both conceptual and application-based test questions.

RESPOND ONLY IN VALID JSON:
{{
  "exercise_1": {{
    "prompt": "Teaching scenario that tests core understanding",
    "focus": "specific learning objective",
    "test_relevance": "how this connects to typical exam questions"
  }},
  "exercise_2": {{
    "prompt": "Application scenario requiring deeper thinking",
    "focus": "conceptual gap this addresses",
    "test_relevance": "exam skill this develops"
  }},
  "bonus_challenge": {{
    "prompt": "Advanced synthesis question",
    "focus": "high-level connection or application",
    "test_relevance": "advanced test question type"
  }}
}}"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert AI tutor specializing in test preparation and conceptual mastery."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Higher creativity for exercise variety
            max_tokens=700,   # More tokens for richer exercises
            request_timeout=10
        )

        content = response["choices"][0]["message"]["content"].strip()
        logger.info("✅ Exercise generation completed")

        # Clean JSON response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            exercises = json.loads(json_match.group())
        else:
            exercises = json.loads(content)

        return FeynmanExerciseResponse(
            exercise_1=exercises["exercise_1"],
            exercise_2=exercises["exercise_2"],
            bonus_challenge=exercises.get("bonus_challenge")
        )

    except Exception as e:
        logger.exception("❌ Exercise generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Exercise generation service temporarily unavailable")

# -----------------------------
# New: Quick Concept Assessment
# -----------------------------
@router.post("/feynman/quick-assess")
def quick_concept_assessment(request: Request, concept: str, user_explanation: str):
    """Fast concept assessment for immediate feedback"""
    try:
        prompt = f"""Rate this explanation of "{concept}" on a scale of 1-10:
"{user_explanation}"

Respond in this exact format:
Score: X/10
One thing they got right: [specific strength]
One thing to improve: [specific gap]
Quick tip: [actionable advice]"""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            request_timeout=5
        )

        return {"quick_feedback": response["choices"][0]["message"]["content"].strip()}

    except Exception as e:
        logger.exception("❌ Quick assessment failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Quick assessment unavailable")
