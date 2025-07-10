# ENHANCED QUIZ MODULE WITH ADVANCED TUTORING FEATURES

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
# Enhanced Models and Enums
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
    FILL_IN_BLANK = "fill_in_blank"
    SHORT_ANSWER = "short_answer"
    MATCHING = "matching"
    ORDERING = "ordering"

class LearningObjective(str, Enum):
    KNOWLEDGE = "knowledge"        # Remember facts
    COMPREHENSION = "comprehension" # Understand concepts
    APPLICATION = "application"    # Apply knowledge
    ANALYSIS = "analysis"         # Break down information
    SYNTHESIS = "synthesis"       # Create new understanding
    EVALUATION = "evaluation"     # Make judgments

@dataclass
class LearningGap:
    concept: str
    importance: float  # 0-1 scale
    prerequisite: bool
    description: str
    suggested_resources: List[str]

@dataclass
class QuizMetrics:
    generation_time: float
    context_fetch_time: float
    question_count: int
    difficulty_distribution: Dict[str, int]
    learning_objectives_covered: List[str]

# Enhanced Models
class QuizRequest(BaseModel):
    content: str = Field(..., min_length=10, description="Learning content to create quiz from")
    difficulty: Optional[DifficultyLevel] = DifficultyLevel.MEDIUM
    question_types: Optional[List[QuestionType]] = [QuestionType.MULTIPLE_CHOICE]
    user_id: Optional[str] = None
    learning_objectives: Optional[List[LearningObjective]] = None
    focus_areas: Optional[List[str]] = None
    time_limit_minutes: Optional[int] = Field(None, ge=1, le=120)
    adaptive_difficulty: bool = True
    include_prerequisites: bool = True
    max_questions: int = Field(12, ge=5, le=25)
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Content must be at least 10 characters long')
        return v.strip()

class QuizQuestion(BaseModel):
    id: int
    type: QuestionType
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel
    learning_objective: LearningObjective
    concept_tags: List[str]
    estimated_time_seconds: int
    hints: List[str] = []
    follow_up_resources: List[str] = []
    prerequisite_concepts: List[str] = []

class QuizResponse(BaseModel):
    quiz_id: str
    questions: List[QuizQuestion]
    metadata: Dict[str, Any]
    learning_gaps_identified: List[Dict[str, Any]]
    estimated_completion_time: int
    adaptive_recommendations: Dict[str, Any]
    prerequisite_check: Dict[str, Any]

# -----------------------------
# Enhanced Context Cache with Performance Improvements
# -----------------------------

class ContextCache:
    def __init__(self, ttl_minutes: int = 5):
        self.cache = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.last_cleanup = datetime.now()
    
    def _cleanup_expired(self):
        now = datetime.now()
        if now - self.last_cleanup > timedelta(minutes=1):
            expired_keys = [
                key for key, (timestamp, _) in self.cache.items()
                if now - timestamp > self.ttl
            ]
            for key in expired_keys:
                del self.cache[key]
            self.last_cleanup = now
    
    async def get_context(self, user_id: str) -> Dict[str, Any]:
        self._cleanup_expired()
        now = datetime.now()
        
        if user_id in self.cache:
            timestamp, cached_value = self.cache[user_id]
            if now - timestamp < self.ttl:
                return cached_value
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{CONTEXT_API}/api/context/cache?user_id={user_id}",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response.raise_for_status()
                    context = await response.json()
                    self.cache[user_id] = (now, context)
                    return context
        except Exception as e:
            print(f"❌ Failed to fetch context for user {user_id}: {e}")
            return {}

context_cache = ContextCache()

# -----------------------------
# Content Analysis and Gap Detection
# -----------------------------

class ContentAnalyzer:
    @staticmethod
    def extract_key_concepts(content: str) -> List[str]:
        """Extract key concepts from content using NLP techniques"""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        # Remove common words and duplicates
        concepts = list(set([w for w in words if len(w) > 3]))
        return concepts[:15]  # Limit to top 15 concepts
    
    @staticmethod
    def identify_learning_gaps(content: str, user_context: Dict[str, Any]) -> List[LearningGap]:
        """Identify potential learning gaps based on content and user context"""
        gaps = []
        
        # Check for missing prerequisites
        weak_areas = user_context.get('weak_areas', [])
        concepts = ContentAnalyzer.extract_key_concepts(content)
        
        for concept in concepts:
            if any(weak in concept.lower() for weak in weak_areas):
                gaps.append(LearningGap(
                    concept=concept,
                    importance=0.8,
                    prerequisite=True,
                    description=f"Potential weakness in {concept} identified",
                    suggested_resources=[f"Review {concept} fundamentals"]
                ))
        
        return gaps
    
    @staticmethod
    def estimate_content_difficulty(content: str) -> DifficultyLevel:
        """Estimate content difficulty based on various factors"""
        # Simple heuristics (can be enhanced with ML models)
        word_count = len(content.split())
        complex_words = len(re.findall(r'\b\w{10,}\b', content))
        technical_terms = len(re.findall(r'\b[A-Z]{2,}\b', content))
        
        difficulty_score = (complex_words / word_count) * 100 + (technical_terms / word_count) * 50
        
        if difficulty_score < 5:
            return DifficultyLevel.BEGINNER
        elif difficulty_score < 10:
            return DifficultyLevel.EASY
        elif difficulty_score < 20:
            return DifficultyLevel.MEDIUM
        elif difficulty_score < 30:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.EXPERT

# -----------------------------
# Enhanced Question Generation
# -----------------------------

class QuestionGenerator:
    @staticmethod
    def build_enhanced_prompt(
        content: str,
        difficulty: DifficultyLevel,
        question_types: List[QuestionType],
        context: Dict[str, Any],
        learning_objectives: List[LearningObjective],
        max_questions: int
    ) -> str:
        
        # Enhanced system message
        system_context = f"""
        You are an expert educational content creator specializing in adaptive learning.
        
        Create {max_questions} high-quality quiz questions that:
        1. Test deep understanding, not just memorization
        2. Include varying difficulty levels to challenge the student appropriately
        3. Cover multiple learning objectives (knowledge, comprehension, application, analysis)
        4. Include helpful explanations that teach additional concepts
        5. Provide hints that guide learning without giving away answers
        6. Identify prerequisite concepts students should know
        7. Suggest follow-up resources for deeper learning
        
        Content Analysis:
        - Key concepts: {ContentAnalyzer.extract_key_concepts(content)}
        - Estimated difficulty: {ContentAnalyzer.estimate_content_difficulty(content)}
        """
        
        # User context integration
        user_context = f"""
        Student Profile:
        - Current topic: {context.get('current_topic', 'General')}
        - Weak areas: {', '.join(context.get('weak_areas', []))}
        - Strong areas: {', '.join(context.get('strong_areas', []))}
        - Learning goals: {', '.join(context.get('user_goals', []))}
        - Previous performance: {context.get('average_score', 'Unknown')}
        - Concepts needing review: {', '.join(context.get('review_queue', []))}
        """
        
        # Question specifications
        question_specs = f"""
        Question Requirements:
        - Difficulty level: {difficulty.value}
        - Question types: {[qt.value for qt in question_types]}
        - Learning objectives to cover: {[lo.value for lo in learning_objectives] if learning_objectives else 'All applicable'}
        - Focus on practical application and critical thinking
        - Include real-world examples where possible
        - Ensure questions build upon each other progressively
        """
        
        # Output format specification
        output_format = """
        Return ONLY a valid JSON array with this exact structure:
        [
          {
            "id": 1,
            "type": "multiple_choice",
            "question": "Detailed, thought-provoking question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "Option A",
            "explanation": "Comprehensive explanation that teaches additional concepts",
            "difficulty": "medium",
            "learning_objective": "application",
            "concept_tags": ["concept1", "concept2"],
            "estimated_time_seconds": 90,
            "hints": ["Helpful hint 1", "Helpful hint 2"],
            "follow_up_resources": ["Resource suggestion 1"],
            "prerequisite_concepts": ["prerequisite1", "prerequisite2"]
          }
        ]
        """
        
        return f"{system_context}\n\n{user_context}\n\n{question_specs}\n\n{output_format}"
    
    @staticmethod
    async def generate_questions(
        content: str,
        difficulty: DifficultyLevel,
        question_types: List[QuestionType],
        context: Dict[str, Any],
        learning_objectives: List[LearningObjective],
        max_questions: int
    ) -> List[QuizQuestion]:
        
        start_time = datetime.now()
        
        try:
            prompt = QuestionGenerator.build_enhanced_prompt(
                content, difficulty, question_types, context, learning_objectives, max_questions
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Upgraded to GPT-4 for better quality
                    messages=[
                        {"role": "system", "content": "You are an expert educational content creator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=3000  # Increased for more detailed responses
                )
            )
            
            raw_content = response["choices"][0]["message"]["content"].strip()
            
            # Clean up response
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = "\n".join(raw_content.splitlines()[1:-1])
            
            parsed_questions = json.loads(raw_content)
            
            # Validate and enhance questions
            questions = []
            for i, q_data in enumerate(parsed_questions):
                try:
                    # Ensure all required fields have defaults
                    q_data.setdefault('id', i + 1)
                    q_data.setdefault('difficulty', difficulty.value)
                    q_data.setdefault('learning_objective', 'comprehension')
                    q_data.setdefault('concept_tags', [])
                    q_data.setdefault('estimated_time_seconds', 60)
                    q_data.setdefault('hints', [])
                    q_data.setdefault('follow_up_resources', [])
                    q_data.setdefault('prerequisite_concepts', [])
                    
                    # Convert boolean answers to strings
                    if isinstance(q_data.get('correct_answer'), bool):
                        q_data['correct_answer'] = str(q_data['correct_answer'])
                    
                    questions.append(QuizQuestion(**q_data))
                except Exception as e:
                    print(f"❌ Error processing question {i}: {e}")
                    continue
            
            generation_time = (datetime.now() - start_time).total_seconds()
            print(f"✅ Generated {len(questions)} questions in {generation_time:.2f}s")
            
            return questions
            
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

# -----------------------------
# Enhanced Logging and Analytics
# -----------------------------

class LearningAnalytics:
    @staticmethod
    async def log_enhanced_learning_event(
        topic: str,
        questions: List[QuizQuestion],
        user_id: Optional[str],
        metrics: QuizMetrics,
        learning_gaps: List[LearningGap]
    ):
        if not user_id:
            return
        
        try:
            payload = {
                "source": "enhanced_quiz",
                "user_id": user_id,
                "current_topic": topic,
                "timestamp": datetime.now().isoformat(),
                "learning_event": {
                    "concept": topic,
                    "phase": "adaptive_quiz",
                    "question_count": len(questions),
                    "difficulty_distribution": metrics.difficulty_distribution,
                    "learning_objectives": metrics.learning_objectives_covered,
                    "estimated_time": sum(q.estimated_time_seconds for q in questions),
                    "concepts_covered": list(set(
                        tag for q in questions for tag in q.concept_tags
                    )),
                    "gaps_identified": [asdict(gap) for gap in learning_gaps],
                    "performance_metrics": asdict(metrics)
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{CONTEXT_API}/api/context/update",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    print(f"📊 Analytics logged: {response.status}")
                    
        except Exception as e:
            print(f"❌ Failed to log analytics: {e}")

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

def build_adaptive_recommendations(
    questions: List[QuizQuestion],
    learning_gaps: List[LearningGap],
    user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Build adaptive learning recommendations"""
    
    difficulty_counts = {}
    for q in questions:
        difficulty_counts[q.difficulty.value] = difficulty_counts.get(q.difficulty.value, 0) + 1
    
    return {
        "next_session_difficulty": "medium",  # Can be enhanced with ML
        "focus_areas": [gap.concept for gap in learning_gaps if gap.importance > 0.7],
        "study_time_recommendation": sum(q.estimated_time_seconds for q in questions) + 300,
        "prerequisite_review_needed": any(gap.prerequisite for gap in learning_gaps),
        "learning_path_suggestions": [
            f"Review {gap.concept}" for gap in learning_gaps[:3]
        ]
    }

# -----------------------------
# Enhanced API Routes
# -----------------------------

@router.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    return {
        "status": "healthy",
        "module": "enhanced_quiz_tutor",
        "version": "2.0.0",
        "features": [
            "adaptive_difficulty",
            "learning_gap_detection",
            "personalized_content",
            "advanced_analytics",
            "prerequisite_checking"
        ],
        "timestamp": datetime.now().isoformat()
    }

@router.post("/generate", response_model=QuizResponse)
async def create_enhanced_quiz(
    req: QuizRequest,
    request: Request,
    background_tasks: BackgroundTasks
):
    """Generate an enhanced, personalized quiz"""
    
    print(f"🚀 Creating enhanced quiz for content: {req.content[:100]}...")
    start_time = datetime.now()
    
    # Extract user identification
    user_id = extract_user_id(request, req)
    print(f"🔍 User ID: {user_id}")
    
    # Fetch user context with timing
    context_start = datetime.now()
    user_context = await context_cache.get_context(user_id) if user_id else {}
    context_fetch_time = (datetime.now() - context_start).total_seconds()
    
    # Analyze content and detect learning gaps
    learning_gaps = ContentAnalyzer.identify_learning_gaps(req.content, user_context)
    content_difficulty = ContentAnalyzer.estimate_content_difficulty(req.content)
    
    # Adjust difficulty based on user context and content analysis
    target_difficulty = req.difficulty
    if req.adaptive_difficulty and user_context.get('average_score'):
        avg_score = user_context.get('average_score', 0.5)
        if avg_score > 0.8:
            target_difficulty = DifficultyLevel.HARD
        elif avg_score < 0.6:
            target_difficulty = DifficultyLevel.EASY
    
    # Set default learning objectives if not provided
    learning_objectives = req.learning_objectives or [
        LearningObjective.COMPREHENSION,
        LearningObjective.APPLICATION,
        LearningObjective.ANALYSIS
    ]
    
    # Generate enhanced questions
    questions = await QuestionGenerator.generate_questions(
        content=req.content,
        difficulty=target_difficulty,
        question_types=req.question_types,
        context=user_context,
        learning_objectives=learning_objectives,
        max_questions=req.max_questions
    )
    
    # Calculate metrics
    total_time = (datetime.now() - start_time).total_seconds()
    difficulty_distribution = {}
    for q in questions:
        difficulty_distribution[q.difficulty.value] = difficulty_distribution.get(q.difficulty.value, 0) + 1
    
    metrics = QuizMetrics(
        generation_time=total_time,
        context_fetch_time=context_fetch_time,
        question_count=len(questions),
        difficulty_distribution=difficulty_distribution,
        learning_objectives_covered=[obj.value for obj in learning_objectives]
    )
    
    # Build adaptive recommendations
    recommendations = build_adaptive_recommendations(questions, learning_gaps, user_context)
    
    # Create quiz response
    quiz_id = f"enhanced_quiz_{uuid.uuid4().hex[:8]}"
    
    quiz_response = QuizResponse(
        quiz_id=quiz_id,
        questions=questions,
        metadata={
            "generation_time": total_time,
            "content_difficulty": content_difficulty.value,
            "target_difficulty": target_difficulty.value,
            "user_context_available": bool(user_context),
            "adaptive_mode": req.adaptive_difficulty,
            "creation_timestamp": datetime.now().isoformat()
        },
        learning_gaps_identified=[asdict(gap) for gap in learning_gaps],
        estimated_completion_time=sum(q.estimated_time_seconds for q in questions),
        adaptive_recommendations=recommendations,
        prerequisite_check={
            "required": [gap.concept for gap in learning_gaps if gap.prerequisite],
            "recommended": [gap.concept for gap in learning_gaps if not gap.prerequisite]
        }
    )
    
    # Log analytics in background
    current_topic = user_context.get('current_topic', 'General')
    background_tasks.add_task(
        LearningAnalytics.log_enhanced_learning_event,
        current_topic, questions, user_id, metrics, learning_gaps
    )
    
    print(f"✅ Enhanced quiz created in {total_time:.2f}s with {len(questions)} questions")
    return quiz_response

@router.get("/analytics/{quiz_id}")
async def get_quiz_analytics(quiz_id: str):
    """Get analytics for a specific quiz"""
    # This would typically fetch from a database
    return {
        "quiz_id": quiz_id,
        "message": "Analytics endpoint ready for implementation",
        "suggested_metrics": [
            "completion_rate",
            "average_score",
            "time_spent",
            "difficulty_progression",
            "learning_objectives_met"
        ]
    }

@router.post("/feedback")
async def submit_quiz_feedback(
    quiz_id: str,
    feedback: Dict[str, Any],
    user_id: Optional[str] = None
):
    """Submit feedback to improve future quiz generation"""
    print(f"📝 Received feedback for quiz {quiz_id}: {feedback}")
    
    # Process feedback to improve recommendations
    # This would typically update user preferences and model parameters
    
    return {
        "status": "feedback_received",
        "quiz_id": quiz_id,
        "message": "Thank you for your feedback. This will help improve future quizzes."
    }
