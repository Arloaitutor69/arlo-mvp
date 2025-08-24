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
    difficulty: Literal["beginner", "easy", "medium", "hard", "expert"]

# Simplified Response Model
class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]
    total_questions: int
    estimated_time_minutes: int

# --- JSON Schema for structured outputs --- #
QUIZ_SCHEMA = {
    "name": "quiz_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 7,
                "maxItems": 15,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "integer",
                            "minimum": 1
                        },
                        "type": {
                            "type": "string",
                            "enum": ["multiple_choice", "true_false", "short_answer"]
                        },
                        "question": {
                            "type": "string",
                            "minLength": 10
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 1
                            },
                            "minItems": 2,
                            "maxItems": 5
                        },
                        "correct_answer": {
                            "type": "string",
                            "minLength": 1
                        },
                        "explanation": {
                            "type": "string",
                            "minLength": 20
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["beginner", "easy", "medium", "hard", "expert"]
                        }
                    },
                    "required": ["id", "type", "question", "correct_answer", "explanation", "difficulty"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["questions"],
        "additionalProperties": False
    }
}

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

EXAMPLE HIGH-QUALITY QUESTIONS:

Multiple Choice Example:
{
    "id": 1,
    "type": "multiple_choice",
    "question": "Which of the following correctly describes the role of the Electron Transport Chain in cellular respiration?",
    "options": [
        "It breaks down glucose into pyruvate in the cytoplasm",
        "It generates oxygen for use in the Krebs Cycle",
        "It transfers electrons to pump protons and produce ATP",
        "It converts carbon dioxide into glucose for energy"
    ],
    "correct_answer": "It transfers electrons to pump protons and produce ATP",
    "explanation": "The Electron Transport Chain uses electrons from NADH and FADH₂ to pump protons across the membrane, creating a gradient that drives ATP synthesis.",
    "difficulty": "medium"
}

Application-Based Example:
{
    "id": 2,
    "type": "multiple_choice",
    "question": "What happens to ATP production if oxygen is unavailable during cellular respiration?",
    "options": [
        "The cell increases use of the Krebs Cycle",
        "ATP production continues normally in the mitochondria",
        "The Electron Transport Chain halts, and glycolysis becomes the main source of ATP",
        "Oxygen is replaced by glucose as the final electron acceptor"
    ],
    "correct_answer": "The Electron Transport Chain halts, and glycolysis becomes the main source of ATP",
    "explanation": "Without oxygen, the ETC stops functioning because oxygen is the final electron acceptor. The cell must rely on glycolysis, which is far less efficient at producing ATP.",
    "difficulty": "hard"
}

Analysis Example:
{
    "id": 3,
    "type": "multiple_choice", 
    "question": "A poison disables enzymes in the Krebs Cycle. What is the most likely consequence for ATP production in the cell?",
    "options": [
        "The cell will produce more ATP through glycolysis to compensate",
        "The cell's total ATP production will decrease significantly",
        "The Electron Transport Chain will function normally using glucose alone",
        "The cell will increase carbon dioxide output due to faster glucose breakdown"
    ],
    "correct_answer": "The cell's total ATP production will decrease significantly",
    "explanation": "The Krebs Cycle is a key source of NADH and FADH₂, which fuel the Electron Transport Chain. Disabling it reduces the input to ETC, lowering total ATP output.",
    "difficulty": "expert"
}

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
# Optimized Question Generation
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
                    "json_schema": QUIZ_SCHEMA
                },
                reasoning_effort="low"
            )

            # Parse the guaranteed valid JSON response
            raw_content = response.choices[0].message.content
            parsed_data = json.loads(raw_content)
            
            # Convert to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_data["questions"]):
                try:
                    # Handle options based on question type
                    if q_data["type"] in ["true_false", "short_answer"]:
                        q_data["options"] = None
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            return questions
            
        except json.JSONDecodeError as e:
            # This should never happen with structured outputs, but just in case
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse response as JSON: {str(e)}"
            )
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
