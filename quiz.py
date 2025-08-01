# OPTIMIZED QUIZ MODULE FOR MAXIMUM OUTPUT

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Literal, Union, Optional, Dict, Any
import uuid
import os
import json
import openai
import requests
import asyncio
import aiohttp
from datetime import datetime, timedelta
import hashlib
import re
from dataclasses import dataclass, asdict
from enum import Enum

# Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
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
    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel

# Simplified Response Model
class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]
    total_questions: int
    estimated_time_minutes: int

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
                    timeout=aiohttp.ClientTimeout(total=2)  # Reduced timeout for speed
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return {}
        except Exception as e:
            print(f"❌ Context fetch failed: {e}")
            return {}

context_cache = ContextCache()

# -----------------------------
# Optimized Question Generation
# -----------------------------

class QuestionGenerator:
    @staticmethod
    def build_optimized_prompt(
        content: str,
        difficulty: DifficultyLevel,
        question_types: List[QuestionType],
        max_questions: int,
        user_weak_areas: List[str] = None
    ) -> str:
        
        # Ultra-concise prompt focused on output maximization
        question_type_str = ", ".join([qt.value for qt in question_types])
        weak_areas_str = f" Focus extra attention on: {', '.join(user_weak_areas[:3])}" if user_weak_areas else ""
        
        prompt = f"""Create exactly {max_questions} quiz questions directly from this content.

CONTENT:
{content}

CRITICAL JSON FORMATTING RULES:
- Output ONLY a valid JSON array of objects
- NO markdown formatting, NO code blocks, NO extra text
- Use double quotes for all strings, NO single quotes
- Ensure proper comma placement between objects and properties
- End with closing square bracket ]

REQUIREMENTS:
- Test understanding, not just memorization based on teaching content and weak areas {weak_areas_str}

OUTPUT FORMAT - COPY THIS EXACT STRUCTURE:
[
{{
  "id": 1,
  "type": "multiple_choice",
  "question": "Which of the following correctly describes the role of the Electron Transport Chain in cellular respiration?",
  "options": ["It breaks down glucose into pyruvate in the cytoplasm", "It generates oxygen for use in the Krebs Cycle", "It transfers electrons to pump protons and produce ATP", "It converts carbon dioxide into glucose for energy"],
  "correct_answer": "It transfers electrons to pump protons and produce ATP",
  "explanation": "The Electron Transport Chain uses electrons from NADH and FADH2 to pump protons across the membrane, creating a gradient that drives ATP synthesis.",
  "difficulty": "medium"
}},
{{
  "id": 2,
  "type": "multiple_choice",
  "question": "What happens to the Electron Transport Chain when oxygen is unavailable in the cell?",
  "options": ["It continues functioning using carbon dioxide as the final electron acceptor", "It stops functioning and the cell must rely on glycolysis alone for ATP production", "It switches to using glucose directly as an electron acceptor", "It increases its rate to compensate for the lack of oxygen"],
  "correct_answer": "It stops functioning and the cell must rely on glycolysis alone for ATP production",
  "explanation": "Oxygen acts as the final electron acceptor in the ETC. Without it, the ETC halts because electrons cannot be passed along the chain, forcing the cell to shift to anaerobic processes like glycolysis for energy production.",
  "difficulty": "hard"
}},
{{
  "id": 3,
  "type": "multiple_choice",
  "question": "A poison disables enzymes in the Krebs Cycle. What is the most likely consequence for ATP production in the cell?",
  "options": ["The cell will produce more ATP through glycolysis to compensate", "The cell's total ATP production will decrease significantly", "The Electron Transport Chain will function normally using glucose alone", "The cell will increase carbon dioxide output due to faster glucose breakdown"],
  "correct_answer": "The cell's total ATP production will decrease significantly",
  "explanation": "The Krebs Cycle is a key source of NADH and FADH2, which fuel the Electron Transport Chain. Disabling it reduces the input to ETC, lowering total ATP output.",
  "difficulty": "expert"
}}
]

JSON VALIDATION REQUIREMENTS:
- Each question object must have ALL 6 fields: id, type, question, options, correct_answer, explanation, difficulty
- Use "multiple_choice" for type (exactly as shown)
- Options array must contain exactly 4 strings
- No trailing commas after last object or property
- Difficulty must be one of: "beginner", "easy", "medium", "hard", "expert"

QUALITY STANDARDS: 
1. Test deep understanding and conceptual reasoning for active recall
2. Include varying difficulty levels within the specified range
3. Cover multiple learning objectives (knowledge, comprehension, application, analysis)
4. Write clear, detailed explanations that teach additional concepts
5. Provide 4 distinct, plausible options that test understanding
6. Focus on application and analysis rather than simple memorization

""" 
        
        return prompt
    
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
            
            prompt = QuestionGenerator.build_optimized_prompt(
                content, difficulty, question_types, max_questions, weak_areas
            )
            
            # Speed optimizations for GPT-3.5-turbo
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are an expert quiz generator. Create exactly {max_questions} high-quality multiple-choice questions optimized for active recall. CRITICAL: Output ONLY valid JSON array format. NO markdown, NO explanatory text, NO code blocks. Start with [ and end with ]. Use proper JSON syntax with double quotes and correct comma placement."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,  # Reduced for more consistent output
                    max_tokens=3500,  # Optimized token limit
                    top_p=0.8,        # Reduced for faster generation
                    frequency_penalty=0.1,  # Slight penalty to avoid repetition
                    presence_penalty=0.1    # Encourage variety
                )
            )
            
            raw_content = response["choices"][0]["message"]["content"].strip()
            
            # Aggressive cleaning of response for speed
            if raw_content.startswith("```"):
                # Fast regex-based cleaning
                raw_content = re.sub(r'^```[a-z]*\n', '', raw_content)
                raw_content = re.sub(r'\n```
            
            # Enhanced JSON parsing with comprehensive error handling
            try:
                parsed_questions = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"❌ Initial JSON parse error: {e}")
                print(f"Error at line {getattr(e, 'lineno', 'unknown')}, column {getattr(e, 'colno', 'unknown')}")
                print(f"Raw content preview: {raw_content[:500]}...")
                
                # Comprehensive JSON cleaning and repair
                try:
                    # Step 1: Remove any markdown formatting
                    cleaned_content = re.sub(r'^```[a-z]*\n?', '', raw_content)
                    cleaned_content = re.sub(r'\n?```
            
            # Fast validation and conversion to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_questions[:max_questions]):  # Limit to requested count
                try:
                    # Ensure required fields exist
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Validate question type - only MCQ and short answer allowed
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = QuestionType.MULTIPLE_CHOICE.value
                    
                    # Ensure MCQ questions have options
                    if q_data.get('type') == 'multiple_choice' and not q_data.get('options'):
                        print(f"⚠️  MCQ question {i} missing options, skipping")
                        continue
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # Quality check
            if len(questions) < max_questions * 0.7:  # At least 70% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print(f"Content length: {len(content)}")
            print(f"Max questions: {max_questions}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified logging with reduced timeout"""
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
                timeout=aiohttp.ClientTimeout(total=3)  # Reduced timeout
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
    
    # Get user context quickly with reduced timeout
    user_id = extract_user_id(request, req)
    
    # Run context fetch and question generation concurrently for speed
    context_task = context_cache.get_context(user_id) if user_id else asyncio.sleep(0)
    
    user_context = await context_task if user_id else {}
    
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
    
    # Log in background (non-blocking)
    if user_id:
        background_tasks.add_task(
            log_quiz_creation,
            user_id,
            user_context.get('current_topic', 'General'),
            len(questions)
        )
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Quiz created in {len(questions)} questions")
    
    return quiz_response, '', cleaned_content)
                    
                    # Step 2: Fix common quote issues
                    cleaned_content = cleaned_content.replace("'", '"')
                    cleaned_content = cleaned_content.replace('"', '"').replace('"', '"')  # Smart quotes
                    
                    # Step 3: Fix boolean and null values
                    cleaned_content = re.sub(r'\bTrue\b', 'true', cleaned_content)
                    cleaned_content = re.sub(r'\bFalse\b', 'false', cleaned_content)
                    cleaned_content = re.sub(r'\bNone\b', 'null', cleaned_content)
                    
                    # Step 4: Fix trailing commas (common JSON error)
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas before }
                    cleaned_content = re.sub(r',\s*]', ']', cleaned_content)  # Remove trailing commas before ]
                    
                    # Step 5: Ensure proper array formatting
                    cleaned_content = cleaned_content.strip()
                    if not cleaned_content.startswith('['):
                        cleaned_content = '[' + cleaned_content
                    if not cleaned_content.endswith(']'):
                        cleaned_content = cleaned_content + ']'
                    
                    # Step 6: Fix missing commas between objects (major cause of parsing errors)
                    # Look for }{ patterns and add comma
                    cleaned_content = re.sub(r'}\s*{', '},{', cleaned_content)
                    
                    print(f"🔧 Attempting to parse cleaned JSON...")
                    parsed_questions = json.loads(cleaned_content)
                    print(f"✅ Successfully parsed after cleaning")
                    
                except json.JSONDecodeError as e2:
                    print(f"❌ JSON repair failed: {e2}")
                    print(f"Cleaned content preview: {cleaned_content[:500]}...")
                    
                    # Last resort: try to extract individual question objects
                    try:
                        print("🆘 Attempting emergency question extraction...")
                        question_pattern = r'\{\s*"id"[^}]+\}'
                        matches = re.findall(question_pattern, raw_content, re.DOTALL)
                        
                        if matches:
                            emergency_questions = []
                            for i, match in enumerate(matches[:max_questions]):
                                try:
                                    # Clean individual question object
                                    clean_match = match.replace("'", '"')
                                    clean_match = re.sub(r',\s*}', '}', clean_match)
                                    question_obj = json.loads(clean_match)
                                    emergency_questions.append(question_obj)
                                except:
                                    continue
                            
                            if emergency_questions:
                                parsed_questions = emergency_questions
                                print(f"🚑 Emergency extraction successful: {len(emergency_questions)} questions")
                            else:
                                raise HTTPException(
                                    status_code=500, 
                                    detail=f"JSON parsing failed completely. Original error: {str(e)}"
                                )
                        else:
                            raise HTTPException(
                                status_code=500, 
                                detail=f"No valid question objects found. Original error: {str(e)}"
                            )
                    except Exception as e3:
                        print(f"❌ Emergency extraction failed: {e3}")
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Complete JSON parsing failure. Errors: {str(e)}, {str(e2)}, {str(e3)}"
                        )
            
            # Fast validation and conversion to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_questions[:max_questions]):  # Limit to requested count
                try:
                    # Ensure required fields exist
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Validate question type - only MCQ and short answer allowed
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = QuestionType.MULTIPLE_CHOICE.value
                    
                    # Ensure MCQ questions have options
                    if q_data.get('type') == 'multiple_choice' and not q_data.get('options'):
                        print(f"⚠️  MCQ question {i} missing options, skipping")
                        continue
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # Quality check
            if len(questions) < max_questions * 0.7:  # At least 70% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print(f"Content length: {len(content)}")
            print(f"Max questions: {max_questions}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified logging with reduced timeout"""
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
                timeout=aiohttp.ClientTimeout(total=3)  # Reduced timeout
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
    
    # Get user context quickly with reduced timeout
    user_id = extract_user_id(request, req)
    
    # Run context fetch and question generation concurrently for speed
    context_task = context_cache.get_context(user_id) if user_id else asyncio.sleep(0)
    
    user_context = await context_task if user_id else {}
    
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
    
    # Log in background (non-blocking)
    if user_id:
        background_tasks.add_task(
            log_quiz_creation,
            user_id,
            user_context.get('current_topic', 'General'),
            len(questions)
        )
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Quiz created in {len(questions)} questions")
    
    return quiz_response, '', raw_content)
            
            # Enhanced JSON parsing with comprehensive error handling
            try:
                parsed_questions = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"❌ Initial JSON parse error: {e}")
                print(f"Error at line {getattr(e, 'lineno', 'unknown')}, column {getattr(e, 'colno', 'unknown')}")
                print(f"Raw content preview: {raw_content[:500]}...")
                
                # Comprehensive JSON cleaning and repair
                try:
                    # Step 1: Remove any markdown formatting
                    cleaned_content = re.sub(r'^```[a-z]*\n?', '', raw_content)
                    cleaned_content = re.sub(r'\n?```
            
            # Fast validation and conversion to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_questions[:max_questions]):  # Limit to requested count
                try:
                    # Ensure required fields exist
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Validate question type - only MCQ and short answer allowed
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = QuestionType.MULTIPLE_CHOICE.value
                    
                    # Ensure MCQ questions have options
                    if q_data.get('type') == 'multiple_choice' and not q_data.get('options'):
                        print(f"⚠️  MCQ question {i} missing options, skipping")
                        continue
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # Quality check
            if len(questions) < max_questions * 0.7:  # At least 70% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print(f"Content length: {len(content)}")
            print(f"Max questions: {max_questions}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified logging with reduced timeout"""
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
                timeout=aiohttp.ClientTimeout(total=3)  # Reduced timeout
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
    
    # Get user context quickly with reduced timeout
    user_id = extract_user_id(request, req)
    
    # Run context fetch and question generation concurrently for speed
    context_task = context_cache.get_context(user_id) if user_id else asyncio.sleep(0)
    
    user_context = await context_task if user_id else {}
    
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
    
    # Log in background (non-blocking)
    if user_id:
        background_tasks.add_task(
            log_quiz_creation,
            user_id,
            user_context.get('current_topic', 'General'),
            len(questions)
        )
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Quiz created in {len(questions)} questions")
    
    return quiz_response, '', cleaned_content)
                    
                    # Step 2: Fix common quote issues
                    cleaned_content = cleaned_content.replace("'", '"')
                    cleaned_content = cleaned_content.replace('"', '"').replace('"', '"')  # Smart quotes
                    
                    # Step 3: Fix boolean and null values
                    cleaned_content = re.sub(r'\bTrue\b', 'true', cleaned_content)
                    cleaned_content = re.sub(r'\bFalse\b', 'false', cleaned_content)
                    cleaned_content = re.sub(r'\bNone\b', 'null', cleaned_content)
                    
                    # Step 4: Fix trailing commas (common JSON error)
                    cleaned_content = re.sub(r',\s*}', '}', cleaned_content)  # Remove trailing commas before }
                    cleaned_content = re.sub(r',\s*]', ']', cleaned_content)  # Remove trailing commas before ]
                    
                    # Step 5: Ensure proper array formatting
                    cleaned_content = cleaned_content.strip()
                    if not cleaned_content.startswith('['):
                        cleaned_content = '[' + cleaned_content
                    if not cleaned_content.endswith(']'):
                        cleaned_content = cleaned_content + ']'
                    
                    # Step 6: Fix missing commas between objects (major cause of parsing errors)
                    # Look for }{ patterns and add comma
                    cleaned_content = re.sub(r'}\s*{', '},{', cleaned_content)
                    
                    print(f"🔧 Attempting to parse cleaned JSON...")
                    parsed_questions = json.loads(cleaned_content)
                    print(f"✅ Successfully parsed after cleaning")
                    
                except json.JSONDecodeError as e2:
                    print(f"❌ JSON repair failed: {e2}")
                    print(f"Cleaned content preview: {cleaned_content[:500]}...")
                    
                    # Last resort: try to extract individual question objects
                    try:
                        print("🆘 Attempting emergency question extraction...")
                        question_pattern = r'\{\s*"id"[^}]+\}'
                        matches = re.findall(question_pattern, raw_content, re.DOTALL)
                        
                        if matches:
                            emergency_questions = []
                            for i, match in enumerate(matches[:max_questions]):
                                try:
                                    # Clean individual question object
                                    clean_match = match.replace("'", '"')
                                    clean_match = re.sub(r',\s*}', '}', clean_match)
                                    question_obj = json.loads(clean_match)
                                    emergency_questions.append(question_obj)
                                except:
                                    continue
                            
                            if emergency_questions:
                                parsed_questions = emergency_questions
                                print(f"🚑 Emergency extraction successful: {len(emergency_questions)} questions")
                            else:
                                raise HTTPException(
                                    status_code=500, 
                                    detail=f"JSON parsing failed completely. Original error: {str(e)}"
                                )
                        else:
                            raise HTTPException(
                                status_code=500, 
                                detail=f"No valid question objects found. Original error: {str(e)}"
                            )
                    except Exception as e3:
                        print(f"❌ Emergency extraction failed: {e3}")
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Complete JSON parsing failure. Errors: {str(e)}, {str(e2)}, {str(e3)}"
                        )
            
            # Fast validation and conversion to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_questions[:max_questions]):  # Limit to requested count
                try:
                    # Ensure required fields exist
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Validate question type - only MCQ and short answer allowed
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = QuestionType.MULTIPLE_CHOICE.value
                    
                    # Ensure MCQ questions have options
                    if q_data.get('type') == 'multiple_choice' and not q_data.get('options'):
                        print(f"⚠️  MCQ question {i} missing options, skipping")
                        continue
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # Quality check
            if len(questions) < max_questions * 0.7:  # At least 70% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            print(f"Content length: {len(content)}")
            print(f"Max questions: {max_questions}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified logging with reduced timeout"""
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
                timeout=aiohttp.ClientTimeout(total=3)  # Reduced timeout
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
    
    # Get user context quickly with reduced timeout
    user_id = extract_user_id(request, req)
    
    # Run context fetch and question generation concurrently for speed
    context_task = context_cache.get_context(user_id) if user_id else asyncio.sleep(0)
    
    user_context = await context_task if user_id else {}
    
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
    
    # Log in background (non-blocking)
    if user_id:
        background_tasks.add_task(
            log_quiz_creation,
            user_id,
            user_context.get('current_topic', 'General'),
            len(questions)
        )
    
    total_time = (datetime.now() - start_time).total_seconds()
    print(f"✅ Quiz created in {len(questions)} questions")
    
    return quiz_response
