from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Literal, Tuple
import os
import json
import requests
import logging
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import re

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

# --- System Prompts --- #
FEYNMAN_EXERCISE_SYSTEM_PROMPT = """You are an expert AI tutor creating Feynman-style teaching exercises for conceptual mastery to consolidate teaching material. 

Create exactly 3 conceptual teaching questions that:
- create questions that test the material and can be answered based on what student learned in teaching content 
- stick to the style of question in examples - ie how and why questions 
- do not explicity reference teaching content in question phrasing, enable student to think and make own connections 
- Test deep understanding, not just memorization
- Help students explain concepts in their own words and recall material learned earlier 
- if the question has multiple part, ensure that each part is lettered ie how did a) something cause and b) something effect. 

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas.

Return exactly 3 questions that encourage conceptual understanding."""

FEYNMAN_ASSESSMENT_SYSTEM_PROMPT = """You are an expert AI tutor assessing a student's conceptual explanation using the Feynman Technique.

Assess the student's explanation and provide:
1. A mastery score out of 100 (be precise and fair)
2. What they did well (specific strengths)
3. Gaps in understanding (specific details that were incorrect or needed elaboration)
    a) you must identify the concept that was missed or vague and provide a clear simple concise explanation of this concept 
4. remain encouraging and never comment on spelling, phrasing etc, only raw content. 

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas."""

# --- JSON examples --- #
EXERCISE_ASSISTANT_EXAMPLE_JSON_1 = """{
  "questions": [
    "a) Why did European exploration expand so rapidly in the late 15th century, b) and what made this timing significant compared to earlier periods?",
    "How did the principles of mercantilism influence a) the goals and b) outcomes of European exploration and colonization?",
    "How did the Columbian Exchange fundamentally transform both European and non-European societies — economically, culturally, and biologically?"
  ]
}"""

EXERCISE_ASSISTANT_EXAMPLE_JSON_2 = """{
  "questions": [
    "a) What is the main function of the cell membrane and b) how does selective permeability work?",
    "a) How do mitochondria produce ATP and b) why is this process essential for cellular life?",
    "a) What role does the nucleus play in controlling cellular activities and b) how does it protect genetic information?"
  ]
}"""

ASSESSMENT_ASSISTANT_EXAMPLE_JSON_1 = """{
  "mastery_score": 48,
  "what_went_well": [
    "Recognized that European nations were seeking new land and wealth",
    "Noted that ships were a key factor in enabling exploration"
  ],
  "gaps_in_understanding": [
    "The response left out why the 15th century marked a turning point — events like the Ottoman Empire's control of land routes, and the spread of Renaissance curiosity and navigation",
    "Motivations for exploration were described too simply, focusing only on wealth, when in reality they also included spreading Christianity and competing with rival nations such as Spain and Portugal.",
    "Saying 'people got rich' was vague, since the real economic forces were the Columbian Exchange (transfer of crops, animals, and diseases), mercantilism (nations building wealth through colonies and trade), and monarchs sponsoring voyages to increase their power.",
    "The impact on indigenous societies was overlooked — exploration brought devastating diseases, forced labor, and cultural imperialism where European languages, religions, and systems replaced native traditions."
  ]
}"""

ASSESSMENT_ASSISTANT_EXAMPLE_JSON_2 = """{
  "mastery_score": 85,
  "what_went_well": [
    "Clearly explained the concept of selective permeability with accurate terminology",
    "Used an effective analogy comparing the cell membrane to a security guard",
    "Demonstrated understanding of the relationship between structure and function"
  ],
  "gaps_in_understanding": [
    "The answer mentioned selective permeability but didn't expand on the role of transport proteins in allowing ions or water to slip through, or carrier proteins which move specific molecules",
    "The structure of the phospholipid bilayer was missing - hydrophilic heads face outward while hydrophobic tails tuck inward, creating a barrier that works like a bouncer."
  ]
}"""

# --- Helper utilities --- #
def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _question_valid(question: str) -> Tuple[bool, Optional[str]]:
    if not isinstance(question, str) or not question.strip():
        return False, "missing or invalid question"
    if len(question) < 20:
        return False, f"question too short ({len(question)} chars)"
    return True, None

def _assessment_valid(assessment: dict) -> Tuple[bool, Optional[str]]:
    if not isinstance(assessment.get("mastery_score"), int):
        return False, "missing or invalid mastery_score"
    if not (0 <= assessment["mastery_score"] <= 100):
        return False, f"mastery_score out of range: {assessment['mastery_score']}"
    if not isinstance(assessment.get("what_went_well"), list) or len(assessment["what_went_well"]) < 1:
        return False, "missing or invalid what_went_well array"
    if not isinstance(assessment.get("gaps_in_understanding"), list) or len(assessment["gaps_in_understanding"]) < 1:
        return False, "missing or invalid gaps_in_understanding array"
    return True, None

def _sanitize_content(raw: str) -> str:
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    return s

def _validate_and_sanitize_questions(questions: List[str]) -> Tuple[bool, Optional[str], List[str]]:
    sanitized = []
    for i, q in enumerate(questions):
        ok, reason = _question_valid(q)
        if not ok:
            return False, f"question {i} invalid: {reason}", questions
        sanitized_question = _sanitize_content(q)
        sanitized.append(sanitized_question)
    return True, None, sanitized

def _validate_and_sanitize_assessment(assessment: dict) -> Tuple[bool, Optional[str], dict]:
    ok, reason = _assessment_valid(assessment)
    if not ok:
        return False, reason, assessment
    
    sanitized_well = [_sanitize_content(item) for item in assessment["what_went_well"]]
    sanitized_gaps = [_sanitize_content(item) for item in assessment["gaps_in_understanding"]]
    
    sanitized_assessment = {
        "mastery_score": assessment["mastery_score"],
        "what_went_well": sanitized_well,
        "gaps_in_understanding": sanitized_gaps
    }
    
    return True, None, sanitized_assessment

# --- OpenAI call wrapper --- #
def _call_model_and_get_parsed_exercise(input_messages, max_tokens=2000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=FeynmanExerciseResponse,
        reasoning={"effort": "low"},
        instructions="Generate exactly 3 conceptual questions that test deep understanding using the Feynman technique.",
        max_output_tokens=max_tokens,
    )

def _call_model_and_get_parsed_assessment(input_messages, max_tokens=3000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=FeynmanAssessmentResponse,
        reasoning={"effort": "low"},
        instructions="Provide detailed assessment of student explanation with specific feedback on strengths and gaps.",
        max_output_tokens=max_tokens,
    )

# -----------------------------
# Enhanced Feynman Exercise Generator
# -----------------------------
@router.post("/feynman/exercises", response_model=FeynmanExerciseResponse)
def generate_feynman_exercises(request: Request, payload: FeynmanExerciseRequest):
    logger.info("🎯 Generating Feynman exercises")

    try:
        user_id = extract_user_id(request, payload)
        context = get_cached_context(user_id)
        
        user_prompt = f"""TEACHING CONTENT:
"{payload.teaching_content}"

Create exactly 3 conceptual teaching questions relevant to this material."""

        # Messages
        input_messages = [
            {"role": "system", "content": FEYNMAN_EXERCISE_SYSTEM_PROMPT},
            {"role": "assistant", "content": EXERCISE_ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": EXERCISE_ASSISTANT_EXAMPLE_JSON_2},
            {"role": "user", "content": user_prompt},
        ]

        # First attempt
        response = _call_model_and_get_parsed_exercise(input_messages)

        if getattr(response, "output_parsed", None) is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
            }
            response = _call_model_and_get_parsed_exercise(input_messages + [retry_msg])
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        questions = response.output_parsed.questions

        # Ensure exactly 3 questions
        if len(questions) != 3:
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: Must have exactly 3 questions. Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed_exercise(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Question count invalid ({len(questions)}). Retry failed.")
            questions = response_retry.output_parsed.questions
            if len(questions) != 3:
                raise HTTPException(status_code=500, detail=f"Question count invalid after retry ({len(questions)}).")

        # Validate + sanitize
        valid, reason, sanitized_questions = _validate_and_sanitize_questions(questions)
        if not valid:
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: Last output failed validation ({reason}). Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed_exercise(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry failed.")
            questions = response_retry.output_parsed.questions
            valid2, reason2, sanitized_questions2 = _validate_and_sanitize_questions(questions)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_questions = sanitized_questions2

        logger.info("✅ Exercise generation completed")
        
        return FeynmanExerciseResponse(questions=sanitized_questions)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Exercise generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Exercise generation service temporarily unavailable")

# -----------------------------
# Enhanced Feynman Assessment
# -----------------------------
@router.post("/feynman/assess", response_model=FeynmanAssessmentResponse)
def assess_feynman_teaching(request: Request, payload: FeynmanAssessmentRequest):
    logger.info("🎓 Assessing Feynman teaching")

    try:
        user_id = extract_user_id(request, payload)
        context = get_cached_context(user_id)
        
        user_prompt = f"""QUESTION: "{payload.question}"

STUDENT'S EXPLANATION:
"{payload.user_explanation}"

Assess this student explanation and provide a detailed evaluation."""

        # Messages
        input_messages = [
            {"role": "system", "content": FEYNMAN_ASSESSMENT_SYSTEM_PROMPT},
            {"role": "assistant", "content": ASSESSMENT_ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": ASSESSMENT_ASSISTANT_EXAMPLE_JSON_2},
            {"role": "user", "content": user_prompt},
        ]

        # First attempt
        response = _call_model_and_get_parsed_assessment(input_messages)

        if getattr(response, "output_parsed", None) is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
            }
            response = _call_model_and_get_parsed_assessment(input_messages + [retry_msg])
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        assessment_data = {
            "mastery_score": response.output_parsed.mastery_score,
            "what_went_well": response.output_parsed.what_went_well,
            "gaps_in_understanding": response.output_parsed.gaps_in_understanding
        }

        # Validate + sanitize
        valid, reason, sanitized_assessment = _validate_and_sanitize_assessment(assessment_data)
        if not valid:
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: Last output failed validation ({reason}). Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed_assessment(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry failed.")
            assessment_data = {
                "mastery_score": response_retry.output_parsed.mastery_score,
                "what_went_well": response_retry.output_parsed.what_went_well,
                "gaps_in_understanding": response_retry.output_parsed.gaps_in_understanding
            }
            valid2, reason2, sanitized_assessment2 = _validate_and_sanitize_assessment(assessment_data)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_assessment = sanitized_assessment2

        logger.info("✅ Assessment completed")

        return FeynmanAssessmentResponse(
            mastery_score=sanitized_assessment["mastery_score"],
            what_went_well=sanitized_assessment["what_went_well"],
            gaps_in_understanding=sanitized_assessment["gaps_in_understanding"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Assessment failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Assessment service temporarily unavailable")
