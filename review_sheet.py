# OPTIMIZED REVIEW SHEET MODULE WITH PERFORMANCE IMPROVEMENTS

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai
import os
import json
import requests
from datetime import datetime, timedelta
from threading import Thread
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

app = FastAPI()
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for non-blocking operations
executor = ThreadPoolExecutor(max_workers=4)

# ---------------------------
# Enhanced In-memory Context Cache
# ---------------------------
context_cache = {}
context_ttl = timedelta(minutes=8)  # Slightly longer TTL for better cache hits
last_cleanup = datetime.now()

def cleanup_expired_cache():
    """Remove expired entries from cache"""
    global last_cleanup
    now = datetime.now()
    if now - last_cleanup > timedelta(minutes=10):
        expired_keys = [
            key for key, (timestamp, _) in context_cache.items()
            if now - timestamp > context_ttl
        ]
        for key in expired_keys:
            del context_cache[key]
        last_cleanup = now
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

async def fetch_context_async(user_id: str) -> dict:
    """Async context fetching with timeout and retry"""
    for attempt in range(2):
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{CONTEXT_API_BASE}/api/context/cache?user_id={user_id}") as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as e:
            logger.warning(f"Async context fetch attempt {attempt + 1} failed for user {user_id}: {e}")
            if attempt == 0:  # Only delay before second attempt
                await asyncio.sleep(0.5)
            continue
    
    logger.error(f"All async context fetch attempts failed for user {user_id}")
    return {}

def get_cached_context(user_id: str) -> dict:
    """Get context with intelligent caching, background refresh, and retry logic"""
    cleanup_expired_cache()
    now = datetime.now()
    
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        # If data is still fresh, return immediately
        if now - timestamp <= context_ttl:
            return cached_value
        # If data is stale but exists, return it and refresh in background
        Thread(target=lambda: asyncio.run(refresh_context_background(user_id))).start()
        return cached_value
    
    # No cache exists, fetch synchronously for first request with retry
    for attempt in range(2):
        try:
            response = requests.get(f"{CONTEXT_API_BASE}/api/context/cache?user_id={user_id}", timeout=5)
            response.raise_for_status()
            context = response.json()
            context_cache[user_id] = (now, context)
            logger.info(f"Successfully fetched context for user {user_id} on attempt {attempt + 1}")
            return context
        except Exception as e:
            logger.warning(f"Context fetch attempt {attempt + 1} failed for user {user_id}: {e}")
            if attempt == 0:  # Only delay before second attempt
                import time
                time.sleep(0.5)
            continue
    
    logger.error(f"All context fetch attempts failed for user {user_id}")
    return {}

async def refresh_context_background(user_id: str):
    """Background context refresh"""
    context = await fetch_context_async(user_id)
    if context:
        context_cache[user_id] = (datetime.now(), context)

# ---------------------------
# Pydantic Models
# ---------------------------
class ReviewRequest(BaseModel):
    user_id: Optional[str] = None

class ReviewSheet(BaseModel):
    summary: str
    memorization_facts: List[str]
    weak_areas: List[str]
    major_topics: List[str]
    study_tips: List[str]  # New field for personalized tips

# ---------------------------
# Extract user_id (optimized)
# ---------------------------
def extract_user_id(request: Request, data: ReviewRequest) -> str:
    # Check most common sources first for performance
    if hasattr(request.state, "user") and request.state.user and "sub" in request.state.user:
        return request.state.user["sub"]
    
    user_id_header = request.headers.get("x-user-id")
    if user_id_header:
        return user_id_header
    
    if data.user_id:
        return data.user_id
    
    raise HTTPException(status_code=400, detail="Missing user_id in request")

# ---------------------------
# Context Processing (NEW)
# ---------------------------
def process_context_for_review(context: dict) -> dict:
    """Pre-process context to extract key information for review generation"""
    processed = {
        "recent_topics": [],
        "struggle_areas": [],
        "key_facts": [],
        "learning_goals": [],
        "session_summary": "",
        "time_spent": {},
        "difficulty_levels": {}
    }
    
    # Extract learning history (most recent first)
    learning_history = context.get("learning_history", [])
    if learning_history:
        # Sort by timestamp if available, otherwise take last 5 entries
        recent_entries = learning_history[-5:] if len(learning_history) > 5 else learning_history
        
        for entry in recent_entries:
            if isinstance(entry, dict):
                topic = entry.get("topic", "")
                if topic:
                    processed["recent_topics"].append(topic)
                
                # Extract struggle indicators
                if entry.get("difficulty") == "hard" or entry.get("attempts", 0) > 2:
                    processed["struggle_areas"].append(topic)
                
                # Extract key facts
                facts = entry.get("facts", [])
                if facts:
                    processed["key_facts"].extend(facts[:3])  # Limit to prevent overload
    
    # Extract emphasized facts
    emphasized_facts = context.get("emphasized_facts", [])
    processed["key_facts"].extend(emphasized_facts[:5])
    
    # Extract weak areas
    weak_areas = context.get("weak_areas", [])
    processed["struggle_areas"].extend(weak_areas)
    
    # Extract user goals
    user_goals = context.get("user_goals", [])
    processed["learning_goals"] = user_goals[:3]  # Limit for focus
    
    # Remove duplicates
    processed["recent_topics"] = list(set(processed["recent_topics"]))
    processed["struggle_areas"] = list(set(processed["struggle_areas"]))
    processed["key_facts"] = list(set(processed["key_facts"]))
    
    return processed

# ---------------------------
# Optimized GPT API Call
# ---------------------------
def call_gpt_optimized(prompt: str) -> str:
    """Optimized GPT call with better parameters for review generation"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are Arlo, an expert learning coach specializing in memory consolidation and spaced repetition. Generate concise, actionable bedtime review sheets that optimize long-term retention."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower for more consistent output
            max_tokens=800,   # Reduced for faster response
            top_p=0.9,        # Focus on high-probability tokens
            frequency_penalty=0.1,  # Reduce repetition
            presence_penalty=0.1    # Encourage diverse content
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT API call failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# ---------------------------
# Enhanced Prompt Generator
# ---------------------------
def build_optimized_review_prompt(processed_context: dict) -> str:
    """Build a focused, optimized prompt for better review generation"""
    
    recent_topics = processed_context.get("recent_topics", [])
    struggle_areas = processed_context.get("struggle_areas", [])
    key_facts = processed_context.get("key_facts", [])
    learning_goals = processed_context.get("learning_goals", [])
    
    prompt = f"""Generate a bedtime review sheet for optimal memory consolidation.

RECENT STUDY SESSION:
Topics covered: {', '.join(recent_topics[:5]) if recent_topics else 'General study session'}
Key facts learned: {', '.join(key_facts[:8]) if key_facts else 'Various concepts'}
Areas of difficulty: {', '.join(struggle_areas[:4]) if struggle_areas else 'None identified'}
Learning goals: {', '.join(learning_goals[:3]) if learning_goals else 'General mastery'}

REQUIREMENTS:
1. Summary: 2-3 sentences highlighting main achievements
2. Memorization facts: 3-5 specific facts perfect for bedtime review (prioritize definitions, formulas, key concepts)
3. Major topics: 3-4 main subjects covered
4. Weak areas: 2-3 specific areas needing more practice
5. Study tips: 2-3 personalized recommendations for tomorrow

Focus on information that benefits from sleep consolidation (facts, procedures, connections).

Respond in JSON format:
{{
  "summary": "...",
  "memorization_facts": ["..."],
  "major_topics": ["..."],
  "weak_areas": ["..."],
  "study_tips": ["..."]
}}"""
    
    return prompt

# ---------------------------
# Fallback Response Generator
# ---------------------------
def generate_fallback_response() -> ReviewSheet:
    """Generate a basic review sheet when context is unavailable"""
    return ReviewSheet(
        summary="You've completed a productive study session today. Great work staying committed to your learning goals!",
        memorization_facts=[
            "Consistent daily review improves long-term retention by 60%",
            "Sleep consolidation helps transfer information from short-term to long-term memory"
        ],
        major_topics=["General study session completed"],
        weak_areas=["Consider tracking specific topics for more detailed feedback"],
        study_tips=[
            "Review these facts again tomorrow morning",
            "Focus on active recall techniques in your next session"
        ]
    )

# ---------------------------
# Main Endpoint (Optimized)
# ---------------------------
@router.post("/review-sheet", response_model=ReviewSheet)
async def generate_review_sheet(request: Request, data: ReviewRequest):
    """Generate optimized review sheet with enhanced caching and processing"""
    
    try:
        user_id = extract_user_id(request, data)
        
        # Get context (with intelligent caching)
        context = get_cached_context(user_id)
        
        if not context:
            logger.warning(f"No context available for user {user_id}, using fallback")
            return generate_fallback_response()
        
        # Process context for optimal review generation
        processed_context = process_context_for_review(context)
        
        # Build optimized prompt
        prompt = build_optimized_review_prompt(processed_context)
        
        logger.info(f"Generating review for user {user_id}")
        
        # Call GPT with optimized settings
        raw_output = await asyncio.get_event_loop().run_in_executor(
            executor, call_gpt_optimized, prompt
        )
        
        # Parse and validate response
        try:
            parsed = json.loads(raw_output)
            review_sheet = ReviewSheet(
                summary=parsed.get("summary", "Study session completed successfully."),
                memorization_facts=parsed.get("memorization_facts", [])[:5],  # Limit facts
                major_topics=parsed.get("major_topics", [])[:4],  # Limit topics
                weak_areas=parsed.get("weak_areas", [])[:3],  # Limit weak areas
                study_tips=parsed.get("study_tips", [])[:3]  # Limit tips
            )
            
            logger.info(f"Successfully generated review sheet for user {user_id}")
            return review_sheet
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GPT response: {e}")
            logger.error(f"Raw GPT output: {raw_output}")
            return generate_fallback_response()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating review sheet: {e}")
        return generate_fallback_response()

# ---------------------------
# Health Check Endpoint
# ---------------------------
@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "cache_size": len(context_cache),
        "timestamp": datetime.now().isoformat()
    }

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001, log_level="info")
