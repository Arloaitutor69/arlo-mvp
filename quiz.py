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
                    timeout=aiohttp.ClientTimeout(total=2)  # Reduced timeout
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
        
        # Ultra-concise prompt focused on output maximization with examples
        question_type_str = ", ".join([qt.value for qt in question_types])
        weak_areas_str = f" Focus extra attention on: {', '.join(user_weak_areas[:3])}" if user_weak_areas else ""
        
        prompt = f"""You are a MASTER QUIZ MAKER TUTOR. Create exactly {max_questions} high-quality quiz questions from the provided content.

CONTENT TO ANALYZE:
{content}

REQUIREMENTS:
- Difficulty: {difficulty.value}
- Types: {question_type_str}
- Match the depth and quality of these EXAMPLES{weak_areas_str}

QUALITY EXAMPLES (MATCH THIS STANDARD):

EXAMPLE 1 - Comprehension Level:
{{
  "id": 1,
  "type": "multiple_choice",
  "question": "Which of the following correctly describes the role of the Electron Transport Chain in cellular respiration?",
  "options": ["A. It breaks down glucose into pyruvate in the cytoplasm", "B. It generates oxygen for use in the Krebs Cycle", "C. It transfers electrons to pump protons and produce ATP", "D. It converts carbon dioxide into glucose for energy"],
  "correct_answer": "C",
  "explanation": "The Electron Transport Chain uses electrons from NADH and FADH₂ to pump protons across the mitochondrial membrane, generating a gradient used to make ATP via oxidative phosphorylation.",
  "difficulty": "medium"
}}

EXAMPLE 2 - Application Level:
{{
  "id": 2,
  "type": "multiple_choice",
  "question": "What happens to ATP production if oxygen is unavailable during cellular respiration?",
  "options": ["A. The cell increases use of the Krebs Cycle", "B. ATP production continues normally in the mitochondria", "C. The Electron Transport Chain halts, and glycolysis becomes the main source of ATP", "D. Oxygen is replaced by glucose as the final electron acceptor"],
  "correct_answer": "C",
  "explanation": "Without oxygen, the ETC stops functioning because oxygen is the final electron acceptor. The cell must rely on glycolysis, which is far less efficient at producing ATP.",
  "difficulty": "hard"
}}

EXAMPLE 3 - Analysis/Reasoning Level:
{{
  "id": 3,
  "type": "multiple_choice",
  "question": "A poison disables enzymes in the Krebs Cycle. What is the most likely consequence for ATP production in the cell?",
  "options": ["A. The cell will produce more ATP through glycolysis to compensate", "B. The cell's total ATP production will decrease significantly", "C. The Electron Transport Chain will function normally using glucose alone", "D. The cell will increase carbon dioxide output due to faster glucose breakdown"],
  "correct_answer": "B",
  "explanation": "The Krebs Cycle produces NADH and FADH₂, which are essential for powering the Electron Transport Chain. If it's blocked, the ETC lacks input, reducing ATP production dramatically.",
  "difficulty": "expert"
}}

YOUR TASK:
1. MATCH the quality and depth shown in examples above
2. Test deep understanding, not memorization
3. Include clear, educational explanations
4. Cover multiple cognitive levels (knowledge, comprehension, application, analysis)
5. Make questions that challenge students appropriately

OUTPUT FORMAT (JSON ARRAY ONLY):
[
  {{
    "id": 1,
    "type": "multiple_choice",
    "question": "Your question here?",
    "options": ["A", "B", "C", "D"],
    "correct_answer": "A",
    "explanation": "Educational explanation that teaches additional concepts.",
    "difficulty": "{difficulty.value}"
  }},
  // ... continue for exactly {max_questions} questions
]

Generate all {max_questions} questions matching the example quality now:""" 
        
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
            
            # Use GPT-4 for better quality and faster processing
            model = "gpt-4" if os.getenv("USE_GPT4", "false").lower() == "true" else "gpt-3.5-turbo"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=model, 
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a master quiz maker tutor. Always create exactly {max_questions} high-quality questions. Return ONLY valid JSON array. No markdown, no explanations, just the JSON."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Slightly higher for creativity
                    max_tokens=5000,  # Increased token limit for better responses
                    top_p=0.9,
                    frequency_penalty=0.3,  # Reduce repetition
                    presence_penalty=0.2    # Encourage diverse topics
                )
            )
            
            raw_content = response["choices"][0]["message"]["content"].strip()
            
            # Aggressive cleaning of response - optimized for speed
            if raw_content.startswith("```"):
                # Find JSON boundaries faster
                start = raw_content.find('[')
                end = raw_content.rfind(']') + 1
                if start != -1 and end > start:
                    raw_content = raw_content[start:end]
            
            # Parse JSON with better error handling
            try:
                parsed_questions = json.loads(raw_content)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                # Try common fixes quickly
                fixes = [
                    lambda x: x.replace("'", '"'),
                    lambda x: x.replace('True', 'true').replace('False', 'false'),
                    lambda x: re.sub(r',\s*}', '}', x),  # Remove trailing commas
                    lambda x: re.sub(r',\s*]', ']', x)   # Remove trailing commas in arrays
                ]
                
                for fix in fixes:
                    try:
                        fixed_content = fix(raw_content)
                        parsed_questions = json.loads(fixed_content)
                        break
                    except:
                        continue
                else:
                    raise HTTPException(status_code=500, detail="Failed to parse AI response")
            
            # Validate and convert to QuizQuestion objects - optimized
            questions = []
            for i, q_data in enumerate(parsed_questions[:max_questions]):  # Limit to max_questions
                try:
                    # Ensure required fields with defaults
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    
                    # Quick type validation and correction
                    if q_data.get('type') not in [qt.value for qt in question_types]:
                        q_data['type'] = question_types[0].value
                    
                    # Handle boolean answers quickly
                    if isinstance(q_data.get('correct_answer'), bool):
                        q_data['correct_answer'] = str(q_data['correct_answer'])
                    
                    # Ensure options exist for multiple choice
                    if q_data['type'] == 'multiple_choice' and not q_data.get('options'):
                        print(f"⚠️  Question {i+1} missing options, skipping")
                        continue
                    
                    questions.append(QuizQuestion(**q_data))
                    
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            print(f"✅ Generated {len(questions)} questions (requested: {max_questions})")
            
            # Quality check - ensure we have sufficient questions
            if len(questions) < max_questions * 0.7:  # At least 70% of requested
                print(f"⚠️  Only got {len(questions)} questions, expected {max_questions}")
                # Could implement retry logic here if needed
                
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Simplified Logging (Made Async for Speed)
# -----------------------------

async def log_quiz_creation(user_id: str, topic: str, question_count: int):
    """Simplified async logging"""
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
        
        # Fire and forget logging to avoid blocking
        asyncio.create_task(
            _log_to_api(payload)
        )
                
    except Exception as e:
        print(f"❌ Logging failed: {e}")

async def _log_to_api(payload: dict):
    """Internal logging function"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_API}/api/context/update",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=3)  # Reduced timeout
            ) as response:
                print(f"📊 Quiz logged: {response.status}")
    except Exception as e:
        print(f"❌ API logging failed: {e}")

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
    """Generate optimized quiz with maximum questions and faster response"""
    
    print(f"🚀 Creating quiz: {req.max_questions} questions from {len(req.content)} chars")
    start_time = datetime.now()
    
    # Parallel execution for speed
    user_id = extract_user_id(request, req)
    
    # Start context fetch and question generation concurrently
    context_task = context_cache.get_context(user_id) if user_id else asyncio.create_task(asyncio.sleep(0))
    
    # Get user context
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
        estimated_time_minutes=max(1, estimated_time // 60)  # At least 1 minute
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
    print(f"✅ Quiz created in {total_time:.2f}s: {len(questions)} questions")
    
    return quiz_response
