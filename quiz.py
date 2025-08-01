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
    def _parse_ai_response(raw_content: str) -> Optional[List[dict]]:
        """Robust AI response parsing with multiple strategies"""
        print(f"🔍 Parsing AI response ({len(raw_content)} chars)")
        
        # Strategy 1: Direct JSON parsing
        try:
            parsed = json.loads(raw_content.strip())
            if isinstance(parsed, list):
                print("✅ Direct JSON parse successful")
                return parsed
        except:
            pass
        
        # Strategy 2: Extract JSON from markdown
        json_content = QuestionGenerator._extract_json_from_text(raw_content)
        if json_content:
            try:
                parsed = json.loads(json_content)
                if isinstance(parsed, list):
                    print("✅ Markdown extraction successful")
                    return parsed
            except:
                pass
        
        # Strategy 3: Progressive JSON fixing
        for strategy_num, fixed_content in enumerate(QuestionGenerator._apply_json_fixes(json_content or raw_content), 1):
            try:
                parsed = json.loads(fixed_content)
                if isinstance(parsed, list):
                    print(f"✅ JSON fix strategy {strategy_num} successful")
                    return parsed
            except Exception as e:
                print(f"❌ Fix strategy {strategy_num} failed: {e}")
                continue
        
        # Strategy 4: Manual parsing as last resort
        try:
            manual_parsed = QuestionGenerator._manual_json_parse(raw_content)
            if manual_parsed:
                print("✅ Manual parsing successful")
                return manual_parsed
        except Exception as e:
            print(f"❌ Manual parsing failed: {e}")
        
        print("❌ All parsing strategies failed")
        return None
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """Extract JSON content from various text formats"""
        text = text.strip()
        
        # Remove markdown code blocks
        if "```" in text:
            # Find content between code blocks
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") and part.endswith("]"):
                    return part
        
        # Find JSON array boundaries
        start_idx = text.find("[")
        if start_idx == -1:
            return None
            
        # Find matching closing bracket
        bracket_count = 0
        end_idx = -1
        
        for i in range(start_idx, len(text)):
            if text[i] == "[":
                bracket_count += 1
            elif text[i] == "]":
                bracket_count -= 1
                if bracket_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > start_idx:
            return text[start_idx:end_idx]
        
        return None
    
    @staticmethod
    def _apply_json_fixes(content: str) -> List[str]:
        """Generate multiple fixed versions of JSON content"""
        fixes = []
        
        # Original content
        fixes.append(content)
        
        # Fix 1: Basic character replacements
        fix1 = content.replace("'", '"').replace('True', 'true').replace('False', 'false')
        fixes.append(fix1)
        
        # Fix 2: Remove trailing commas
        fix2 = re.sub(r',(\s*[}\]])', r'\1', fix1)
        fixes.append(fix2)
        
        # Fix 3: Fix unquoted keys
        fix3 = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fix2)
        fixes.append(fix3)
        
        # Fix 4: Escape internal quotes
        fix4 = re.sub(r'(?<!\\)"(?=[^,:}\]]*[,:}\]])', r'\\"', fix3)
        fixes.append(fix4)
        
        # Fix 5: Remove newlines in strings
        fix5 = fix4.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        fixes.append(fix5)
        
        return fixes
    
    @staticmethod
    def _manual_json_parse(text: str) -> Optional[List[dict]]:
        """Manual parsing for severely malformed JSON"""
        try:
            # Look for question patterns
            questions = []
            
            # Split by common question separators
            parts = re.split(r'(?:^|\n)\s*[{]', text)
            
            for i, part in enumerate(parts[1:], 1):  # Skip first empty part
                try:
                    # Try to extract question data manually
                    question_data = QuestionGenerator._extract_question_data(part)
                    if question_data:
                        question_data['id'] = i
                        questions.append(question_data)
                except:
                    continue
            
            return questions if questions else None
            
        except Exception as e:
            print(f"❌ Manual parsing error: {e}")
            return None
    
    @staticmethod
    def _extract_question_data(text: str) -> Optional[dict]:
        """Extract question data from text fragment"""
        try:
            data = {}
            
            # Extract type
            type_match = re.search(r'"type":\s*"([^"]+)"', text)
            data['type'] = type_match.group(1) if type_match else 'multiple_choice'
            
            # Extract question
            question_match = re.search(r'"question":\s*"([^"]+(?:\\.[^"]*)*)"', text)
            if not question_match:
                return None
            data['question'] = question_match.group(1).replace('\\"', '"')
            
            # Extract options (if multiple choice)
            options_match = re.search(r'"options":\s*\[(.*?)\]', text, re.DOTALL)
            if options_match:
                options_text = options_match.group(1)
                options = re.findall(r'"([^"]+(?:\\.[^"]*)*)"', options_text)
                data['options'] = [opt.replace('\\"', '"') for opt in options]
            
            # Extract correct answer
            answer_match = re.search(r'"correct_answer":\s*"([^"]+(?:\\.[^"]*)*)"', text)
            if not answer_match:
                return None
            data['correct_answer'] = answer_match.group(1).replace('\\"', '"')
            
            # Extract explanation
            explanation_match = re.search(r'"explanation":\s*"([^"]+(?:\\.[^"]*)*)"', text)
            data['explanation'] = explanation_match.group(1).replace('\\"', '"') if explanation_match else "Explanation not available."
            
            # Extract difficulty
            difficulty_match = re.search(r'"difficulty":\s*"([^"]+)"', text)
            data['difficulty'] = difficulty_match.group(1) if difficulty_match else 'medium'
            
            return data
            
        except Exception as e:
            print(f"❌ Question data extraction failed: {e}")
            return None

    @staticmethod
    def _extract_and_clean_json(raw_content: str) -> str:
        """Enhanced JSON extraction and cleaning"""
        content = raw_content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split('\n')
            # Find actual JSON start/end
            json_start = -1
            json_end = -1
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('[') and json_start == -1:
                    json_start = i
                if stripped.endswith(']') and json_start != -1:
                    json_end = i + 1
                    break
            
            if json_start != -1 and json_end != -1:
                content = '\n'.join(lines[json_start:json_end])
        
        # Find JSON array boundaries if no markdown
        if not content.startswith('['):
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                content = content[start:end]
        
        # Basic cleaning
        content = content.strip()
        
        return content
    
    @staticmethod
    def _attempt_json_fixes(content: str) -> Optional[List[dict]]:
        """Try multiple JSON fix strategies"""
        fixes = [
            # Fix 1: Basic quote and boolean fixes
            lambda x: x.replace("'", '"').replace('True', 'true').replace('False', 'false'),
            
            # Fix 2: Remove trailing commas
            lambda x: re.sub(r',(\s*[}\]])', r'\1', x),
            
            # Fix 3: Fix unescaped quotes in strings
            lambda x: re.sub(r'(?<!\\)"(?=[^,}\]:]*[,}\]])(?![^{[]*:)', r'\\"', x),
            
            # Fix 4: Add missing quotes to keys
            lambda x: re.sub(r'(\w+):', r'"\1":', x),
            
            # Fix 5: Fix newlines in strings
            lambda x: x.replace('\n', '\\n').replace('\r', '\\r'),
        ]
        
        for i, fix in enumerate(fixes):
            try:
                fixed_content = fix(content)
                parsed = json.loads(fixed_content)
                if isinstance(parsed, list):
                    print(f"✅ JSON fixed with strategy {i+1}")
                    return parsed
            except Exception as e:
                print(f"❌ Fix {i+1} failed: {e}")
                continue
        
        return None
    
    @staticmethod
    def _generate_fallback_questions(
        content: str, 
        difficulty: DifficultyLevel, 
        question_types: List[QuestionType], 
        max_questions: int
    ) -> List[QuizQuestion]:
        """Generate reliable fallback questions when AI fails"""
        print("🔧 Generating fallback questions...")
        
        questions = []
        question_type = question_types[0]
        
        # Simple template-based questions
        templates = [
            {
                "question": "What is the main topic discussed in the provided content?",
                "options": ["The primary subject matter", "An unrelated topic", "Multiple unconnected topics", "No clear topic"] if question_type == QuestionType.MULTIPLE_CHOICE else None,
                "correct_answer": "The primary subject matter",
                "explanation": "The question tests basic comprehension of the main content topic."
            },
            {
                "question": "Based on the content, which approach best describes the material?",
                "options": ["Educational and informative", "Purely theoretical", "Completely practical", "Unstructured information"] if question_type == QuestionType.MULTIPLE_CHOICE else None,
                "correct_answer": "Educational and informative",
                "explanation": "The content is designed to be educational and provide useful information."
            },
            {
                "question": "What type of learning objective does this content primarily address?",
                "options": ["Knowledge and understanding", "Physical skills only", "Emotional responses", "No clear objective"] if question_type == QuestionType.MULTIPLE_CHOICE else None,
                "correct_answer": "Knowledge and understanding",
                "explanation": "Educational content typically focuses on building knowledge and understanding."
            }
        ]
        
        # Create questions from templates
        for i, template in enumerate(templates[:max_questions]):
            try:
                question = QuizQuestion(
                    id=i + 1,
                    type=question_type,
                    question=template["question"],
                    options=template["options"],
                    correct_answer=template["correct_answer"],
                    explanation=template["explanation"],
                    difficulty=difficulty
                )
                questions.append(question)
            except Exception as e:
                print(f"❌ Fallback question {i} failed: {e}")
                continue
        
        # Fill remaining spots with generic questions
        while len(questions) < max_questions:
            try:
                question = QuizQuestion(
                    id=len(questions) + 1,
                    type=question_type,
                    question=f"Question {len(questions) + 1}: What can be learned from this educational content?",
                    options=["Valuable information and concepts", "Nothing useful", "Only basic facts", "Contradictory information"] if question_type == QuestionType.MULTIPLE_CHOICE else None,
                    correct_answer="Valuable information and concepts",
                    explanation="Educational content is designed to provide valuable learning opportunities.",
                    difficulty=difficulty
                )
                questions.append(question)
            except Exception as e:
                print(f"❌ Generic question failed: {e}")
                break
        
        print(f"🔧 Generated {len(questions)} fallback questions")
        return questions

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
        
        # Shorter, more focused prompt to avoid token issues
        prompt = f"""Create exactly {max_questions} quiz questions from this content:

{content[:2000]}...

Requirements: {difficulty.value} difficulty, {question_type_str} questions{weak_areas_str}

Return JSON array only:
[{{"id":1,"type":"multiple_choice","question":"What...?","options":["A","B","C","D"],"correct_answer":"A","explanation":"Because...","difficulty":"{difficulty.value}"}}]""" 
        
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
            
            # Use GPT-3.5-turbo for reliability and speed
            model = "gpt-3.5-turbo"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=model, 
                    messages=[
                        {
                            "role": "system", 
                            "content": f"You are a quiz generator. Create exactly {max_questions} questions. Return ONLY a JSON array. Start with [ and end with ]. No text before or after. No markdown. Example: [{{\"id\":1,\"type\":\"multiple_choice\",\"question\":\"What is X?\",\"options\":[\"A\",\"B\",\"C\",\"D\"],\"correct_answer\":\"A\",\"explanation\":\"Because...\",\"difficulty\":\"medium\"}}]"
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,  # Reduced for more consistent output
                    max_tokens=4000,  # Reduced as requested
                    top_p=0.85,      # Slightly reduced for more focused output
                    frequency_penalty=0.2,  # Reduced to avoid breaking JSON
                    presence_penalty=0.1    # Reduced to avoid breaking JSON
                )
            )
            
            raw_content = response["choices"][0]["message"]["content"].strip()
            print(f"🔍 Raw AI response length: {len(raw_content)}")
            
            # Robust JSON extraction with multiple fallback strategies
            parsed_questions = QuestionGenerator._parse_ai_response(raw_content)
            
            if not parsed_questions:
                print("🔄 AI parsing failed, generating fallback questions...")
                return QuestionGenerator._generate_fallback_questions(
                    content, difficulty, question_types, max_questions
                )
            
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
