from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
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

# Load environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CONTEXT_BASE_URL = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# -----------------------------
# Models
# -----------------------------
class FlashcardRequest(BaseModel):
    content: str
    format: Optional[str] = "Q&A"
    user_id: Optional[str] = None

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

# --- Output schema for structured parsing --- #
class FlashcardResponse(BaseModel):
    flashcards: List[dict]

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
# Prompt Engineering
# -----------------------------
def build_flashcard_prompt(content: str, context: Dict[str, Any]) -> str:
    # Build personalization context
    personalization = _build_personalization_context(context)
    
    return f"""You are a personalized flashcard-generating tutor. Create detailed, optimized flashcards for memory retention and understanding.

CONTENT TO PROCESS - Extract and consolidate key information from what was taught:
{content}

PERSONALIZATION CONTEXT:
{personalization}

GENERATION REQUIREMENTS:
Create exactly 12-20 flashcards focusing on:
- FACTS that need memorization
- DEFINITIONS of key terms and concepts  
- IMPORTANT DETAILS that students commonly forget
- SPECIFIC INFORMATION that benefits from spaced repetition

Prioritize information that can be best memorized using flashcards. Focus on consolidating what was actually taught to the student.

QUALITY STANDARDS:
- Questions should be unambiguous and test understanding, not just recall
- Answers should be complete but concise
- Include examples in answers when they aid understanding
- Use active voice and clear language
- Ensure each card addresses a specific learning objective
- Prioritize information that benefits from spaced repetition

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas.

EXAMPLE FORMAT:
Content: Cell Biology - The cell membrane controls what enters and exits the cell through selective permeability. Mitochondria produce ATP energy. The nucleus contains DNA and controls cell functions.

example Flashcards:
- Question: "What is the main function of the cell membrane?" Answer: "The cell membrane controls what substances can enter and exit the cell through selective permeability, acting like a security guard for the cell."
- Question: "What is the primary function of mitochondria?" Answer: "Mitochondria produce ATP (adenosine triphosphate), which is the cell's main source of energy."
- Question: "What does the nucleus contain and what is its role?" Answer: "The nucleus contains the cell's DNA and serves as the control center, directing all cellular activities and functions."

Create flashcards that help students memorize and understand the key concepts from the teaching content."""

def _build_personalization_context(context: Dict[str, Any]) -> str:
    if not context:
        return "No personalization context available"
    
    return f"""Current Topic: {context.get('current_topic', 'General')}
Learning Goals: {', '.join(context.get('user_goals', [])) or 'Not specified'}
Strong Areas: {', '.join(context.get('strong_areas', [])) or 'Not specified'}
Weak Areas: {', '.join(context.get('weak_areas', [])) or 'Not specified'}
Preferred Learning Style: {context.get('learning_style', 'Not specified')}
Recent Study Sessions: {len(context.get('recent_sessions', []))}
Review Queue Size: {len(context.get('review_queue', []))}"""

# --- JSON examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """{
  "flashcards": [
    {
      "question": "What is the main function of the cell membrane?",
      "answer": "The cell membrane controls what substances can enter and exit the cell through selective permeability, acting like a security guard for the cell."
    },
    {
      "question": "What is the primary function of mitochondria?",
      "answer": "Mitochondria produce ATP (adenosine triphosphate), which is the cell's main source of energy."
    }
  ]
}"""

ASSISTANT_EXAMPLE_JSON_2 = """{
  "flashcards": [
    {
      "question": "What is opportunity cost in economics?",
      "answer": "Opportunity cost is the value of the next best alternative that you give up when making a choice. For example, if you spend $10 on lunch, the opportunity cost is the movie ticket you could have bought instead."
    },
    {
      "question": "What does scarcity mean in economics?",
      "answer": "Scarcity means that resources (money, time, materials, etc.) are limited while our wants and needs are unlimited. This forces us to make choices about how to use our resources."
    }
  ]
}"""

ASSISTANT_EXAMPLE_JSON_3 = """{
  "flashcards": [
    {
      "question": "",
      "answer": ""
    }
  ]
}"""

# --- Helper utilities --- #
def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _flashcard_valid(flashcard: dict) -> (bool, Optional[str]):
    if not isinstance(flashcard.get("question"), str) or not flashcard["question"].strip():
        return False, "missing or invalid question"
    if not isinstance(flashcard.get("answer"), str) or not flashcard["answer"].strip():
        return False, "missing or invalid answer"
    if len(flashcard["question"]) < 10:
        return False, "question too short"
    if len(flashcard["answer"]) < 5:
        return False, "answer too short"
    return True, None

def _sanitize_content(raw: str) -> str:
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    return s

def _validate_and_sanitize_flashcards(flashcards: List[dict]) -> (bool, Optional[str], List[dict]):
    sanitized = []
    for i, card in enumerate(flashcards):
        if not isinstance(card, dict):
            return False, f"flashcard {i} is not a dictionary", flashcards
        ok, reason = _flashcard_valid(card)
        if not ok:
            return False, f"flashcard {i} invalid: {reason}", flashcards
        sanitized_question = _sanitize_content(card["question"])
        sanitized_answer = _sanitize_content(card["answer"])
        sanitized.append({
            "question": sanitized_question,
            "answer": sanitized_answer
        })
    return True, None, sanitized

# --- OpenAI call wrapper --- #
def _call_model_and_get_parsed(input_messages, max_tokens=4000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=FlashcardResponse,
        reasoning={"effort": "low"},
        instructions="Generate flashcards that focus on key facts, definitions, and concepts that benefit from spaced repetition.",
        max_output_tokens=max_tokens,
    )

def generate_flashcards_sync(
    content: str,
    context: Dict[str, Any],
    request: FlashcardRequest
) -> List[Dict[str, Any]]:
    """Synchronous flashcard generation with enhanced AI prompting"""
    
    try:
        # Build prompt
        system_prompt = build_flashcard_prompt(content, context)
        
        # User prompt
        user_prompt = f"Create flashcards for this content: {content}\n\nOutput exactly 12-20 flashcards in valid JSON format."
        
        # Messages
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
            {"role": "user", "content": user_prompt},
        ]

        # First attempt
        response = _call_model_and_get_parsed(input_messages)

        if getattr(response, "output_parsed", None) is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
            }
            response = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        flashcards = response.output_parsed.flashcards

        # Ensure 12–20 flashcards
        if not (12 <= len(flashcards) <= 20):
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: Must have 12-20 flashcards. Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Flashcard count invalid ({len(flashcards)}). Retry failed.")
            flashcards = response_retry.output_parsed.flashcards
            if not (12 <= len(flashcards) <= 20):
                raise HTTPException(status_code=500, detail=f"Flashcard count invalid after retry ({len(flashcards)}).")

        # Validate + sanitize
        valid, reason, sanitized_flashcards = _validate_and_sanitize_flashcards(flashcards)
        if not valid:
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: Last output failed validation ({reason}). Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry failed.")
            flashcards = response_retry.output_parsed.flashcards
            valid2, reason2, sanitized_flashcards2 = _validate_and_sanitize_flashcards(flashcards)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_flashcards = sanitized_flashcards2

        return sanitized_flashcards
        
    except HTTPException:
        raise
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
