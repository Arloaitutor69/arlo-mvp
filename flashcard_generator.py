# flashcards.py

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
import json
import uuid
import requests
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass

# Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Enhanced Models
# -----------------------------
@dataclass
class LearningGap:
    concept: str
    description: str
    priority: str  # "high", "medium", "low"
    prerequisites: List[str]

class FlashcardRequest(BaseModel):
    content: str
    format: Optional[str] = "Q&A"
    user_id: Optional[str] = None
    difficulty_level: Optional[str] = "adaptive"  # "beginner", "intermediate", "advanced", "adaptive"
    focus_areas: Optional[List[str]] = []
    learning_objectives: Optional[List[str]] = []
    exclude_topics: Optional[List[str]] = []

class FlashcardItem(BaseModel):
    id: str
    front: str
    back: str
    difficulty: str
    category: str
    subcategory: Optional[str] = None
    learning_objective: Optional[str] = None
    prerequisite_concepts: Optional[List[str]] = []
    confidence_level: Optional[float] = 0.5
    estimated_time_seconds: Optional[int] = 30
    tags: Optional[List[str]] = []
    explanation: Optional[str] = None  # Additional context/explanation

# Remove FlashcardResponse class - using original format

# -----------------------------
# Enhanced Context Cache
# -----------------------------
context_cache = {}
context_ttl = timedelta(minutes=5)

async def get_cached_context_async(user_id: str) -> Dict[str, Any]:
    """Async version of context fetching for better performance"""
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return cached_value
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{CONTEXT_BASE_URL}/api/context/cache?user_id={user_id}",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    context = await response.json()
                    context_cache[user_id] = (now, context)
                    return context
                else:
                    print(f"❌ Context API returned status {response.status}")
                    return {}
    except Exception as e:
        print("❌ Failed to fetch context:", e)
        return {}

def get_cached_context(user_id: str) -> Dict[str, Any]:
    """Sync wrapper for backwards compatibility"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_cached_context_async(user_id))
    finally:
        loop.close()

# -----------------------------
# Learning Analytics
# -----------------------------
class LearningAnalyzer:
    @staticmethod
    def analyze_content_complexity(content: str) -> Dict[str, Any]:
        """Analyze content to determine complexity and key concepts"""
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Extract potential key concepts (capitalized terms, technical terms)
        key_concepts = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        technical_terms = re.findall(r'\b\w+(?:tion|sion|ment|ness|ity|ism|ology|graphy)\b', content)
        
        complexity_score = min(10, (avg_sentence_length / 15) * 5 + (len(technical_terms) / 10) * 5)
        
        return {
            "complexity_score": complexity_score,
            "key_concepts": list(set(key_concepts[:10])),
            "technical_terms": list(set(technical_terms[:10])),
            "word_count": word_count,
            "estimated_reading_time": word_count / 200  # minutes
        }
    
    @staticmethod
    def identify_learning_gaps(content: str, user_context: Dict[str, Any]) -> List[LearningGap]:
        """Identify potential learning gaps based on content and user context"""
        gaps = []
        weak_areas = user_context.get('weak_areas', [])
        
        # Analyze content for prerequisite concepts
        content_analysis = LearningAnalyzer.analyze_content_complexity(content)
        
        for concept in content_analysis['key_concepts']:
            if any(weak_area.lower() in concept.lower() for weak_area in weak_areas):
                gaps.append(LearningGap(
                    concept=concept,
                    description=f"User has indicated weakness in {concept}",
                    priority="high",
                    prerequisites=[]
                ))
        
        return gaps

# -----------------------------
# Enhanced Prompt Engineering
# -----------------------------
class FlashcardPromptBuilder:
    @staticmethod
    def build_enhanced_prompt(
        content: str,
        context: Dict[str, Any],
        request: FlashcardRequest,
        content_analysis: Dict[str, Any],
        learning_gaps: List[LearningGap]
    ) -> str:
        
        # Determine optimal difficulty based on context and request
        difficulty_level = FlashcardPromptBuilder._determine_difficulty(
            request.difficulty_level, context, content_analysis
        )
        
        # Build personalization context
        personalization = FlashcardPromptBuilder._build_personalization_context(context)
        
        # Build learning gaps context
        gaps_context = FlashcardPromptBuilder._build_gaps_context(learning_gaps)
        
        prompt = f"""
You are a personalized flashcard-generating tutor. Please generate detailed, optimized flashcards for memory retention and understanding.

CONTENT TO PROCESS - Extract and consolidate key information from what was taught:
{content}

PERSONALIZATION CONTEXT:
{personalization}

LEARNING GAPS TO ADDRESS:
{gaps_context}

CONTENT ANALYSIS:
- Key Concepts: {', '.join(content_analysis['key_concepts'])}
- Technical Terms: {', '.join(content_analysis['technical_terms'])}
- Complexity: {difficulty_level}

GENERATION REQUIREMENTS:
Create exactly 10-15 flashcards as a JSON array focusing on:
- FACTS that need memorization
- DEFINITIONS of key terms and concepts  
- IMPORTANT DETAILS that students commonly forget
- SPECIFIC INFORMATION that benefits from spaced repetition

Prioritize information that can be best memorized using flashcards. Focus on consolidating what was actually taught to the student.

JSON FORMAT (return only this array):
[
  {{ "question": "...", "answer": "..." }}
]

QUALITY STANDARDS:
- Questions should be unambiguous and test understanding, not just recall
- Answers should be complete but concise
- Include examples in answers when they aid understanding
- Use active voice and clear language
- Ensure each card addresses a specific learning objective
- Prioritize information that benefits from spaced repetition

Return ONLY the JSON array with no additional text or formatting.
"""
        return prompt
    
    @staticmethod
    def _determine_difficulty(
        requested_difficulty: str,
        context: Dict[str, Any],
        content_analysis: Dict[str, Any]
    ) -> str:
        if requested_difficulty != "adaptive":
            return requested_difficulty
        
        # Adaptive difficulty based on user context and content
        complexity_score = content_analysis['complexity_score']
        user_experience = len(context.get('user_goals', []))
        
        if complexity_score < 3 and user_experience < 2:
            return "beginner"
        elif complexity_score < 7 and user_experience < 4:
            return "intermediate"
        else:
            return "advanced"
    
    @staticmethod
    def _build_personalization_context(context: Dict[str, Any]) -> str:
        if not context:
            return "No personalization context available"
        
        return f"""
Current Topic: {context.get('current_topic', 'General')}
Learning Goals: {', '.join(context.get('user_goals', [])) or 'Not specified'}
Strong Areas: {', '.join(context.get('strong_areas', [])) or 'Not identified'}
Weak Areas: {', '.join(context.get('weak_areas', [])) or 'Not identified'}
Preferred Learning Style: {context.get('learning_style', 'Not specified')}
Recent Study Sessions: {len(context.get('recent_sessions', []))}
Review Queue Size: {len(context.get('review_queue', []))}
"""
    
    @staticmethod
    def _build_gaps_context(gaps: List[LearningGap]) -> str:
        if not gaps:
            return "No specific learning gaps identified"
        
        gap_descriptions = []
        for gap in gaps:
            gap_descriptions.append(f"- {gap.concept}: {gap.description} (Priority: {gap.priority})")
        
        return "\n".join(gap_descriptions)

# -----------------------------
# Enhanced Generation Function
# -----------------------------
def generate_flashcards_sync(
    content: str,
    context: Dict[str, Any],
    request: FlashcardRequest
) -> List[Dict[str, Any]]:
    """Synchronous flashcard generation with enhanced AI prompting"""
    
    # Analyze content complexity
    content_analysis = LearningAnalyzer.analyze_content_complexity(content)
    
    # Identify learning gaps
    learning_gaps = LearningAnalyzer.identify_learning_gaps(content, context)
    
    # Build enhanced prompt
    prompt = FlashcardPromptBuilder.build_enhanced_prompt(
        content, context, request, content_analysis, learning_gaps
    )
    
    try:
        messages = [{"role": "user", "content": prompt}]
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            reasoning_effort="low"
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Clean up JSON response
        if raw_content.startswith("```"):
            raw_content = "\n".join(raw_content.strip().splitlines()[1:-1])
        
        cards = json.loads(raw_content)
        
        # Validate and enhance cards
        enhanced_cards = []
        for card in cards:
            if not card.get("question") or not card.get("answer"):
                continue
            
            # Keep original simple format
            enhanced_card = {
                "question": card["question"],
                "answer": card["answer"]
            }
            enhanced_cards.append(enhanced_card)
        
        return enhanced_cards
        
    except Exception as e:
        print(f"❌ AI generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate flashcards: {str(e)}")

async def generate_flashcards_async(
    content: str,
    context: Dict[str, Any],
    request: FlashcardRequest
) -> List[Dict[str, Any]]:
    """Async wrapper for flashcard generation"""
    return generate_flashcards_sync(content, context, request)

# -----------------------------
# Enhanced Endpoint
# -----------------------------
@router.post("/flashcards")
async def generate_flashcards(request: Request, data: FlashcardRequest):
    """Enhanced flashcard generation endpoint with original output format"""
    
    # Extract user ID
    user_id = extract_user_id(request, data)
    
    # Get user context
    context = await get_cached_context_async(user_id)
    
    # Set parameters from context
    count = 12
    difficulty = "medium"
    topic = context.get("current_topic", "general")
    
    # Generate flashcards
    try:
        raw_cards = await generate_flashcards_async(data.content, context, data)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate flashcards")
    
    # Convert to original FlashcardItem format
    flashcards = []
    questions_summary = []
    
    for card_data in raw_cards[:count]:
        q = card_data.get("question", "No question.")
        a = card_data.get("answer", "No answer.")
        
        flashcards.append(FlashcardItem(
            id=f"card_{uuid.uuid4().hex[:6]}",
            front=q,
            back=a,
            difficulty=difficulty,
            category=topic
        ))
        questions_summary.append(q)
    
    # Update context with learning event (enhanced)
    await _update_learning_context(user_id, flashcards, topic)
    
    # Return original format
    return {
        "flashcards": flashcards,
        "total_cards": len(flashcards),
        "estimated_time": f"{len(flashcards) * 1.5:.0f} minutes"
    }

# -----------------------------
# Helper Functions
# -----------------------------
def extract_user_id(request: Request, data: FlashcardRequest) -> str:
    """Extract user ID from various sources"""
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

async def _update_learning_context(user_id: str, flashcards: List[FlashcardItem], topic: str):
    """Update learning context with flashcard session data"""
    try:
        questions_summary = [card.front for card in flashcards]
        
        payload = {
            "source": "flashcards",
            "user_id": user_id,
            "current_topic": topic,
            "learning_event": {
                "concept": topic,
                "phase": "flashcards",
                "confidence": 0.5,
                "depth": "shallow",
                "source_summary": "; ".join(questions_summary),
                "repetition_count": 1,
                "review_scheduled": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_BASE_URL}/api/context/update",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status != 200:
                    print(f"❌ Context update failed with status {response.status}")
                    
    except Exception as e:
        print(f"❌ Context update error: {e}")

# Remove the _generate_recommendations function as it's not needed for original format
