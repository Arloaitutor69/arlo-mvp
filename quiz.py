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
        
        prompt = f"""Create a minumum of 7 and max 15 quiz questions from this content.

CONTENT:
{content}

REQUIREMENTS:
- Difficulty: {difficulty.value}
- Types: {question_type_str}
- Test understanding, not just memorization{weak_areas_str}

QUALITY OF OUTPUT: 
1. Test deep understanding, not just memorization
2. Include varying difficulty levels to challenge the student appropriately
3. Cover multiple learning objectives (knowledge, comprehension, application, analysis)
4. Include helpful explanations that teach additional concepts

example of qaulity questions...


    "id": 1,
    "type": "multiple_choice",
    "question": "Which of the following correctly describes the role of the Electron Transport Chain in cellular respiration?",
    "options": [
      "It breaks down glucose into pyruvate in the cytoplasm",
      "It generates oxygen for use in the Krebs Cycle",
      "It transfers electrons to pump protons and produce ATP",
      "It converts carbon dioxide into glucose for energy"
    ],
    "correct_answer": "C",
    "explanation": "The Electron Transport Chain uses electrons from NADH and FADH₂ to pump protons across the membrane, creating a gradient that drives ATP synthesis.",
    "difficulty": "medium"

    "id": 2,
    "type": "multiple_choice",
    "question": "What happens to ATP production if oxygen is unavailable during cellular respiration?",
    "options": [
      "The cell increases use of the Krebs Cycle",
      "ATP production continues normally in the mitochondria",
      "The Electron Transport Chain halts, and glycolysis becomes the main source of ATP",
      "Oxygen is replaced by glucose as the final electron acceptor"
    ],
    "correct_answer": "C",
    "explanation": "Without oxygen, the ETC stops functioning because oxygen is the final electron acceptor. The cell must rely on glycolysis, which is far less efficient at producing ATP.",
    "difficulty": "hard"

    "id": 3,
    "type": "multiple_choice",
    "question": "A poison disables enzymes in the Krebs Cycle. What is the most likely consequence for ATP production in the cell?",
    "options": [
      "The cell will produce more ATP through glycolysis to compensate",
      "The cell's total ATP production will decrease significantly",
      "The Electron Transport Chain will function normally using glucose alone",
      "The cell will increase carbon dioxide output due to faster glucose breakdown"
    ],
    "correct_answer": "B",
    "explanation": "The Krebs Cycle is a key source of NADH and FADH₂, which fuel the Electron Transport Chain. Disabling it reduces the input to ETC, lowering total ATP output.",
    "difficulty": "expert"
 
OUTPUT FORMAT (JSON only, no markdown):
[
  {{
    "id": 1,
    "type": "multiple_choice",
    "question": "Question text here?",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "A",
    "explanation": "Brief explanation why A is correct.",
    "difficulty": "{difficulty.value}"
  }}
]

Generate all {max_questions} questions now:""" 
        
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
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo", 
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a quiz generator. Always create exactly {max_questions} questions. Return only valid JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                    max_tokens=4000,  # Increased token limit
                    top_p=0.9
                )
            )
            
            raw_content = response["choices"][0]["message"]["content"].strip()
            
            # Aggressive cleaning of response
            if raw_content.startswith("```"):
                lines = raw_content.split('\n')
                # Find JSON start and end
                start_idx = 0
                end_idx = len(lines)
                
                for i, line in enumerate(lines):
                    if line.strip().startswith('['):
                        start_idx = i
                        break
                
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().endswith(']'):
                        end_idx = i + 1
                        break
                
                raw_content = '\n'.join(lines[start_idx:end_idx])
            
            # Parse JSON
            try:
                parsed_questions = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"Raw content: {raw_content[:500]}...")
                # Try to fix common JSON issues
                fixed_content = raw_content.replace("'", '"').replace('True', 'true').replace('False', 'false')
                parsed_questions = json.loads(fixed_content)
            
            # Validate and convert to QuizQuestion objects
            questions = []
            for i, q_data in enumerate(parsed_questions):
                try:
                    # Ensure required fields
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Handle boolean answers
                    if isinstance(q_data.get('correct_answer'), bool):
                        q_data['correct_answer'] = str(q_data['correct_answer'])
                    
                    # Validate question type
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = question_types[0].value
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    print(f"Question data: {q_data}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # If we didn't get enough questions, this is a generation issue
            if len(questions) < max_questions * 0.6:  # At least 60% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                # Log the issue but still return what we have
                
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

