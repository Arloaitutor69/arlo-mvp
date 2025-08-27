# OPTIMIZED QUIZ MODULE FOR MAXIMUM OUTPUT WITH GPT-5-NANO STRUCTURED OUTPUTS

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Union, Optional, Dict, Any
from openai import OpenAI
import uuid
import os
import json
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import hashlib
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Simplified Models for Maximum Output
# -----------------------------

class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    SHORT_ANSWER = "short_answer"

class LearningObjective(str, Enum):
    KNOWLEDGE = "knowledge"
    COMPREHENSION = "comprehension"
    APPLICATION = "application"
    ANALYSIS = "analysis"

# Streamlined Models
class QuizRequest(BaseModel):
    content: str = Field(..., min_length=10)
    difficulty: Optional[DifficultyLevel] = DifficultyLevel.MEDIUM
    question_types: Optional[List[QuestionType]] = [QuestionType.MULTIPLE_CHOICE]
    user_id: Optional[str] = None
    max_questions: int = Field(12, ge=8, le=15)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content must be at least 10 characters long')
        return v.strip()

# Simplified Question Model - Only Essential Fields
class QuizQuestion(BaseModel):
    id: int
    type: Literal["multiple_choice", "true_false", "short_answer"]
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str

# Simplified Response Model
class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]
    total_questions: int
    estimated_time_minutes: int

# New response model for GPT-5-nano parsing
class QuizGenerationResponse(BaseModel):
    questions: List[QuizQuestion]

# -----------------------------
# Simplified Context Cache
# -----------------------------

class ContextCache:
    def __init__(self, ttl_minutes: int = 5):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
    
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        if not user_id:
            return {}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{CONTEXT_API}/api/context/cache?user_id={user_id}",
                    timeout=aiohttp.ClientTimeout(total=3)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception as e:
            print(f"❌ Context fetch failed: {e}")
            return {}

context_cache = ContextCache()

# -----------------------------
# Enhanced GPT Prompt with Quality Examples
# -----------------------------

def build_system_prompt() -> str:
    return """You are an expert quiz generator that creates high-quality educational questions. Create exactly 7-15 quiz questions that test deep understanding, not just memorization.

QUALITY REQUIREMENTS:
1. Test understanding, comprehension, and application - not just recall
2. Include varying difficulty levels to appropriately challenge students
3. Cover multiple learning objectives (knowledge, comprehension, application, analysis)
4. Provide helpful explanations that teach additional concepts
5. For multiple choice: create plausible distractors that test common misconceptions

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas.

Remember: Create questions that require students to think, analyze, and apply concepts rather than just memorize facts."""

def build_user_prompt(
    content: str,
    difficulty: DifficultyLevel,
    question_types: List[QuestionType],
    max_questions: int,
    user_weak_areas: List[str] = None
) -> str:
    
    question_type_str = ", ".join([qt.value for qt in question_types])
    weak_areas_str = f" Focus extra attention on: {', '.join(user_weak_areas[:3])}" if user_weak_areas else ""
    
    return f"""Create {max_questions} high-quality quiz questions from this content.

CONTENT:
{content}

REQUIREMENTS:
- Difficulty: {difficulty.value}
- Question Types: {question_type_str}
- Test deep understanding, not just memorization{weak_areas_str}
- Include questions that require analysis and application of concepts
- For multiple choice questions, include the options array
- For true/false questions, set options to null and correct_answer to "true" or "false"
- For short answer questions, set options to null

Create exactly {max_questions} questions that thoroughly test student understanding."""

# -----------------------------
# OpenAI call wrapper - Updated for GPT-5-nano
# -----------------------------
def _call_model_and_get_parsed(input_messages, max_tokens=6000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=QuizGenerationResponse,
        reasoning={"effort": "low"},
        instructions="Generate high-quality quiz questions that test deep understanding and application of concepts, not just memorization.",
        max_output_tokens=max_tokens,
    )

# -----------------------------
# Optimized Question Generation - Updated
# -----------------------------

class QuestionGenerator:
    @staticmethod
    async def generate_questions(
        content: str,
        difficulty: DifficultyLevel,
        question_types: List[QuestionType],
        max_questions: int,
        user_context: Dict[str, Any] = None
    ) -> List[QuizQuestion]:
        
        try:
            # Extract only essential context
            weak_areas = user_context.get('weak_areas', [])[:3] if user_context else []
            
            system_prompt = build_system_prompt()
            user_prompt = build_user_prompt(
                content, difficulty, question_types, max_questions, weak_areas
            )
            
# --- JSON examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """{
  "questions": [
    {
      "id": 1,
      "type": "multiple_choice",
      "question": "Which process directly produces the most ATP during cellular respiration?",
      "options": ["Glycolysis", "Krebs Cycle", "Electron Transport Chain", "Fermentation"],
      "correct_answer": "Electron Transport Chain",
      "explanation": "The Electron Transport Chain produces approximately 32-34 ATP molecules through oxidative phosphorylation, which is far more than glycolysis (2 ATP) or the Krebs Cycle (2 ATP)."
    }
  ]
}"""

ASSISTANT_EXAMPLE_JSON_2 = """{
  "questions": [
    {
      "id": 1,
      "type": "true_false",
      "question": "Photosynthesis and cellular respiration are essentially opposite processes.",
      "options": null,
      "correct_answer": "true",
      "explanation": "Photosynthesis converts CO₂ and H₂O into glucose using light energy, while cellular respiration breaks down glucose into CO₂ and H₂O to release energy. They are complementary opposite processes."
    }
  ]
}"""

ASSISTANT_EXAMPLE_JSON_3 = """{
  "questions": [
    {
      "id": 1,
      "type": "short_answer",
      "question": "What is the main function of mitochondria in cellular respiration?",
      "options": null,
      "correct_answer": "Generate ATP through oxidative phosphorylation",
      "explanation": "Mitochondria are the powerhouses of the cell, using oxygen to break down glucose and produce ATP through the electron transport chain and chemiosmosis."
    }
  ]
}"""

            # Prepare messages with example structure
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
                {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
                {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
                {"role": "user", "content": user_prompt}
            ]

            # First attempt with new GPT-5-nano syntax
            response = _call_model_and_get_parsed(input_messages)

            if getattr(response, "output_parsed", None) is None:
                if hasattr(response, "refusal") and response.refusal:
                    raise HTTPException(status_code=400, detail=response.refusal)
                retry_msg = {
                    "role": "user",
                    "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected quiz questions. Nothing else."
                }
                response = _call_model_and_get_parsed(input_messages + [retry_msg])
                if getattr(response, "output_parsed", None) is None:
                    raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

            questions = response.output_parsed.questions

            # Ensure proper question count
            if not (7 <= len(questions) <= 15):
                retry_msg = {
                    "role": "user",
                    "content": f"Fix JSON only: Must have {max_questions} questions. Return corrected JSON only."
                }
                response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
                if getattr(response_retry, "output_parsed", None) is None:
                    raise HTTPException(status_code=500, detail=f"Question count invalid ({len(questions)}). Retry failed.")
                questions = response_retry.output_parsed.questions
                if not (7 <= len(questions) <= 15):
                    raise HTTPException(status_code=500, detail=f"Question count invalid after retry ({len(questions)}).")

            # Process questions to handle options based on type
            processed_questions = []
            for q in questions:
                # Handle options based on question type
                if q.type in ["true_false", "short_answer"]:
                    q.options = None
                processed_questions.append(q)
            
            print(f"✅ Generated {len(processed_questions)} questions (requested: {max_questions})")
            
            return processed_questions
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print(f"Content length: {len(content)}")
            print(f"Max questions: {max_questions}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified logging"""
    if not user_id:
        return
    
    try:
        payload = {
            "user_id": user_id,
            "event": "quiz_generated",
            "topic": topic,
            "question_count": question_count,
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_API}/api/context/update",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                print(f"📊 Quiz logged: {response.status}")
                
    except Exception as e:
        print(f"❌ Logging failed: {e}")

# -----------------------------
# Utility Functions
# -----------------------------

def extract_user_id(request: Request, req: QuizRequest) -> Optional[str]:
    """Extract user ID from various sources"""
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif req.user_id:
        return req.user_id
    return None

# -----------------------------
# Optimized API Route
# -----------------------------

@router.post("/generate", response_model=QuizResponse)
async def create_quiz(
    req: QuizRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Generate optimized quiz with maximum questions"""
    
    print(f"🚀 Creating quiz: {req.max_questions} questions from {len(req.content)} chars")
    start_time = datetime.now()
    
    # Get user context quickly
    user_id = extract_user_id(request, req)
    user_context = await context_cache.get_context(user_id) if user_id else {}
    
    # Generate questions with optimized approach
    questions = await QuestionGenerator.generate_questions(
        content=req.content,
        difficulty=req.difficulty,
        question_types=req.question_types,
        max_questions=req.max_questions,
        user_context=user_context
    )
    
    # Create response
    quiz_id = f"quiz_{uuid.uuid4().hex[:8]}"
    estimated_time = len(questions) * 90  # 90 seconds per question average
    
    quiz_response = QuizResponse(
        quiz_id=quiz_id,
        questions=questions,
        total_questions=len(questions),
        estimated_time_minutes=estimated_time // 60
    )
    
    # Log in background
    if user_id:
        background_tasks.add_task(
            log_quiz_creation,
            user_id,
            user_context.get('current_topic', 'General'),
            len(questions)
        )
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Quiz created in {total_time:.2f}s: {len(questions)} questions")
    
    return quiz_response
