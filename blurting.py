# OPTIMIZED BLURTING MODULE WITH IMPROVED PERFORMANCE AND CONTENT QUALITY

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
import openai
import os
import json
import requests
import time
from datetime import datetime, timedelta
from threading import Thread
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------- OPTIMIZED CONTEXT CACHE -------------------
context_cache: dict = {}
context_ttl = timedelta(minutes=5)
executor = ThreadPoolExecutor(max_workers=3)  # For non-blocking operations

if os.getenv("ENV") == "dev":
    CONTEXT_BASE = "http://localhost:10000"
else:
    CONTEXT_BASE = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

async def get_cached_context(user_id: str):
    """Async context fetching with improved error handling"""
    now = datetime.now()
    if user_id in context_cache:
        timestamp, cached_value = context_cache[user_id]
        if now - timestamp < context_ttl:
            return {"cached": True, "stale": False, "context": cached_value}
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(f"{CONTEXT_BASE}/api/context/cache?user_id={user_id}") as response:
                if response.status == 200:
                    context = await response.json()
                    context_cache[user_id] = (now, context)
                    return {"cached": False, "stale": False, "context": context}
                else:
                    return {"cached": False, "stale": True, "context": None}
    except Exception as e:
        print(f"❌ Context fetch failed: {e}")
        return {"cached": False, "stale": True, "context": None}

# ------------------- MODELS -------------------
class BlurtingRequest(BaseModel):
    topic: str
    content_summary: Optional[str] = None
    blurted_response: str
    context_prompt: Optional[str] = None
    user_id: Optional[str] = None

class BlurtingResponse(BaseModel):
    feedback: str
    missed_concepts: List[str]
    context_alignment: str

class BlurtingExerciseRequest(BaseModel):
    topic: str
    teaching_block: str
    user_id: Optional[str] = None

class BlurtingExerciseResponse(BaseModel):
    exercise_1: dict
    exercise_2: dict

# ------------------- USER ID EXTRACTION -------------------
def extract_user_id(request: Request, data) -> str:
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif hasattr(data, 'user_id') and data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

# ------------------- ASYNC CONTEXT POSTING -------------------
async def post_learning_event_async(user_id: str, topic: str, missed_concepts: List[str], feedback: str):
    """Async context posting with timeout protection"""
    payload = {
        "source": "blurting",
        "user_id": user_id,
        "current_topic": topic,
        "weak_areas": missed_concepts[:3],
        "review_queue": missed_concepts[:3],
        "learning_event": {
            "concept": topic,
            "phase": "blurting",
            "confidence": 3 if len(missed_concepts) <= 2 else 1 if len(missed_concepts) >= 5 else 2,
            "depth": "deep" if len(missed_concepts) <= 1 else "medium" if len(missed_concepts) <= 3 else "shallow",
            "source_summary": feedback[:150],
            "repetition_count": 1,
            "review_scheduled": True
        }
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            async with session.post(f"{CONTEXT_BASE}/api/context/update", json=payload) as response:
                if response.status == 200:
                    print("✅ Context logged successfully")
                else:
                    print(f"⚠️ Context log returned {response.status}")
    except Exception as e:
        print(f"❌ Context log failed: {e}")

# ------------------- IMPROVED PROMPT GENERATORS -------------------
def generate_evaluation_prompt(topic: str, content_summary: Optional[str], blurted_response: str, context_prompt: Optional[str]) -> str:
    """Enhanced prompt for better evaluation quality"""
    summary_section = f"\n\nKey concepts to evaluate against:\n{content_summary}" if content_summary else ""
    context_section = f"\n\nStudent's learning context (weak areas to focus on):\n{context_prompt}" if context_prompt else ""
    
    return f"""You are an expert educational assessor evaluating a student's blurting exercise on "{topic}".

STUDENT'S RESPONSE:
{blurted_response[:800]}
{summary_section}
{context_section}

EVALUATION CRITERIA:
- Accuracy of core concepts and facts
- Completeness of key information
- Understanding depth vs surface-level recall
- Identification of specific knowledge gaps

Return a JSON object with:
1. "feedback": Constructive feedback (2-3 sentences) that:
   - Acknowledges what they recalled well
   - Points out 1-2 specific missing concepts
   - Suggests focused review areas
   
2. "missed_concepts": Array of 3-5 specific concepts/facts they missed or explained poorly
   
3. "context_alignment": One sentence on how well their response covers the learning objectives

Focus on actionable, specific feedback rather than generic praise. Be encouraging but precise about gaps."""

def generate_exercise_prompt(topic: str, teaching_block: str, context: dict) -> str:
    """Enhanced prompt for better exercise generation"""
    weak_areas = context.get("weak_areas", [])
    weak_areas_text = f"\n\nSTUDENT'S WEAK AREAS (prioritize these): {', '.join(weak_areas)}" if weak_areas else ""
    
    return f"""You are an expert learning scientist designing optimal blurting exercises for "{topic}".

TEACHING CONTENT:
{teaching_block[:1200]}
{weak_areas_text}

BLURTING TECHNIQUE: Students recall information from memory without looking at materials. Most effective for:
- Factual information (dates, names, definitions)
- Sequential processes (steps, procedures)
- Lists and categorizations
- Detailed explanations of concepts

Create 2 distinct exercises targeting different memory retrieval patterns:

EXERCISE 1: Focus on detailed recall (facts, definitions, specific examples)
EXERCISE 2: Focus on process/sequence recall (steps, cause-effect, chronology)

Return JSON format:
{{
  "exercise_1": {{
    "prompt": "Clear, specific instruction for what to recall",
    "focus": "Brief description of what memory skill this targets"
  }},
  "exercise_2": {{
    "prompt": "Clear, specific instruction for what to recall", 
    "focus": "Brief description of what memory skill this targets"
  }}
}}

Make prompts specific and actionable. Avoid generic "explain everything" instructions."""

# ------------------- OPTIMIZED OPENAI CALLS -------------------
async def call_openai_async(messages: List[dict], max_tokens: int = 400, temperature: float = 0.3) -> str:
    """Async OpenAI call with optimized parameters"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=15,
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1    # Encourage variety
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI call failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# ------------------- OPTIMIZED MAIN ENDPOINT -------------------
@router.post("/blurting", response_model=BlurtingResponse)
async def evaluate_blurting(request: Request, data: BlurtingRequest):
    """Optimized blurting evaluation with async processing"""
    try:
        user_id = extract_user_id(request, data)
        
        # Fetch context asynchronously if needed
        if not data.context_prompt:
            context_result = await get_cached_context(user_id)
            if context_result["context"]:
                weak_areas = context_result["context"].get("weak_areas", [])
                data.context_prompt = f"Focus on these concepts: {', '.join(weak_areas[:3])}" if weak_areas else None

        # Generate optimized prompt
        prompt = generate_evaluation_prompt(
            data.topic,
            data.content_summary,
            data.blurted_response,
            data.context_prompt
        )

        # Async OpenAI call
        start_time = time.time()
        content = await call_openai_async([{"role": "user", "content": prompt}])
        print(f"⏱️ OpenAI call: {time.time() - start_time:.2f}s")

        # Parse response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing for malformed JSON
            print("⚠️ JSON parsing failed, attempting cleanup")
            cleaned = content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

        # Validate response structure
        if not all(key in parsed for key in ["feedback", "missed_concepts", "context_alignment"]):
            raise ValueError("Invalid response structure")

        # Fire-and-forget context logging
        asyncio.create_task(post_learning_event_async(
            user_id,
            data.topic,
            parsed["missed_concepts"],
            parsed["feedback"]
        ))

        return BlurtingResponse(**parsed)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        print(f"❌ Blurting evaluation error: {e}")
        raise HTTPException(status_code=500, detail="Evaluation service error")

# ------------------- OPTIMIZED EXERCISE GENERATOR -------------------
@router.post("/blurting/exercises", response_model=BlurtingExerciseResponse)
async def generate_blurting_exercises(request: Request, data: BlurtingExerciseRequest):
    """Optimized exercise generation with better targeting"""
    try:
        user_id = extract_user_id(request, data)
        
        # Get context asynchronously
        context_result = await get_cached_context(user_id)
        context = context_result.get("context", {})

        # Generate improved prompt
        prompt = generate_exercise_prompt(data.topic, data.teaching_block, context)

        # Async OpenAI call with optimal parameters
        messages = [
            {"role": "system", "content": "You are an expert educational content creator specializing in memory retrieval techniques."},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        content = await call_openai_async(messages, max_tokens=500, temperature=0.4)
        print(f"⏱️ Exercise generation: {time.time() - start_time:.2f}s")

        # Parse and validate
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            cleaned = content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

        if not all(key in parsed for key in ["exercise_1", "exercise_2"]):
            raise ValueError("Invalid exercise structure")

        return BlurtingExerciseResponse(
            exercise_1=parsed["exercise_1"],
            exercise_2=parsed["exercise_2"]
        )
        
    except Exception as e:
        print(f"❌ Exercise generation error: {e}")
        raise HTTPException(status_code=500, detail="Exercise generation failed")
