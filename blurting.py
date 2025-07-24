# ENHANCED BLURTING MODULE WITH STRUCTURED FEEDBACK

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
class BlurtingExerciseRequest(BaseModel):
    teaching_block: str
    user_id: Optional[str] = None

class BlurtingExerciseResponse(BaseModel):
    exercise_1: dict
    exercise_2: dict
    exercise_3: dict
    key_concepts: List[str]  # Added to track concepts for evaluation

class BlurtingFeedbackRequest(BaseModel):
    teaching_content: str
    blurted_response: str
    user_id: Optional[str] = None

class BlurtingFeedbackResponse(BaseModel):
    mentioned: List[str]
    partial_mentions: List[str]
    missed: List[str]
    mentioned_count: int
    total_key_concepts: int
    score_fraction: str
    feedback: str

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
async def post_learning_event_async(user_id: str, teaching_content: str, missed_concepts: List[str], feedback: str, score_fraction: str):
    """Async context posting with timeout protection"""
    # Extract topic from teaching content (first sentence or key phrase)
    topic = teaching_content.split('.')[0][:100] if teaching_content else "Blurting Exercise"
    
    # Calculate confidence based on score
    mentioned_count = int(score_fraction.split('/')[0]) if '/' in score_fraction else 0
    total_count = int(score_fraction.split('/')[1]) if '/' in score_fraction else 1
    score_ratio = mentioned_count / total_count if total_count > 0 else 0
    
    payload = {
        "source": "blurting",
        "user_id": user_id,
        "current_topic": topic,
        "weak_areas": missed_concepts[:3],
        "review_queue": missed_concepts[:3],
        "learning_event": {
            "concept": topic,
            "phase": "blurting",
            "confidence": 0.9 if score_ratio >= 0.8 else 0.7 if score_ratio >= 0.6 else 0.5 if score_ratio >= 0.4 else 0.3,
            "depth": "deep" if score_ratio >= 0.8 else "intermediate" if score_ratio >= 0.5 else "shallow",
            "source_summary": feedback[:150],
            "repetition_count": 1,
            "review_scheduled": True,
            "score": score_fraction
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

# ------------------- ENHANCED PROMPT GENERATORS -------------------
def generate_exercise_prompt(teaching_block: str, context: dict) -> str:
    """Enhanced prompt for better exercise generation with key concept extraction"""
    weak_areas = context.get("weak_areas", [])
    weak_areas_text = f"\n\nSTUDENT'S WEAK AREAS (prioritize these): {', '.join(weak_areas)}" if weak_areas else ""
    
    return f"""You are an expert learning scientist designing optimal blurting exercises based on teaching content.

TEACHING CONTENT:
{teaching_block[:1200]}
{weak_areas_text}

BLURTING TECHNIQUE: Students recall information from memory without looking at materials. Most effective for:
- Factual information (dates, names, definitions)
- Sequential processes (steps, procedures)
- Lists and categorizations
- Detailed explanations of concepts

EXAMPLE INPUT:
Teaching Block: "DNA replication is the process by which a cell copies its DNA before cell division. It occurs during the S-phase of the cell cycle, inside the nucleus in eukaryotic cells. The process is semi-conservative, meaning each daughter strand retains one original strand. Replication begins at multiple origins of replication. Helicase unwinds the DNA, while topoisomerase relieves torsional strain. Primase lays down RNA primers to initiate synthesis. DNA Polymerase III extends new strands in the 5' to 3' direction. DNA is synthesized continuously on the leading strand and in short segments (Okazaki fragments) on the lagging strand. DNA Polymerase I replaces RNA primers with DNA, and DNA Ligase seals the fragments."

EXAMPLE OUTPUT:
{{
  "exercise_1": {{
    "prompt": "List all the enzymes involved in DNA replication and describe what each one does.",
    "focus": "Factual recall of specific proteins and their functions"
  }},
  "exercise_2": {{
    "prompt": "Describe the step-by-step process of DNA replication from initiation to completion.",
    "focus": "Sequential process recall and chronological understanding"
  }},
  "exercise_3": {{
    "prompt": "Explain the differences between leading and lagging strand synthesis, including why Okazaki fragments form.",
    "focus": "Conceptual understanding of directional synthesis differences"
  }},
  "key_concepts": [
    "DNA replication",
    "S-phase",
    "Nucleus",
    "Semi-conservative",
    "Origins of replication",
    "Helicase",
    "Topoisomerase",
    "Primase",
    "RNA primers",
    "DNA Polymerase III",
    "5' to 3' direction",
    "Leading strand",
    "Lagging strand",
    "Okazaki fragments",
    "DNA Polymerase I",
    "DNA Ligase"
  ]
}}

Your task:
1. Extract ALL key concepts from the teaching content that students should remember
2. Create 3 distinct exercises targeting different memory retrieval patterns

EXERCISE 1: Focus on detailed recall (facts, definitions, specific examples)
EXERCISE 2: Focus on process/sequence recall (steps, cause-effect, chronology)  
EXERCISE 3: Focus on conceptual understanding (relationships, comparisons, explanations)

Return JSON format exactly like the example above. Make prompts specific and actionable. The key_concepts array should be comprehensive - include every important term, process, fact, or concept from the teaching content."""

def generate_evaluation_prompt(teaching_content: str, blurted_response: str, context_prompt: Optional[str]) -> str:
    """Enhanced prompt following the structured feedback format with examples"""
    context_section = f"\n\nStudent's learning context (weak areas to focus on):\n{context_prompt}" if context_prompt else ""
    
    return f"""You are an expert educational assessor evaluating a student's blurting exercise response.

TEACHING CONTENT (Key concepts to evaluate against):
{teaching_content[:1200]}

STUDENT'S BLURTED RESPONSE:
{blurted_response[:800]}
{context_section}

EVALUATION TASK:
Analyze the student's response and categorize their recall into three groups:

1. **MENTIONED**: Concepts they recalled correctly (exact matches, synonyms, or clearly understood concepts)
2. **PARTIAL MENTIONS**: Concepts they mentioned but got partially wrong, misused, or explained vaguely  
3. **MISSED**: Important concepts from the teaching content that they didn't mention at all

EXAMPLE INPUT:
Teaching Content: "DNA replication is the process by which a cell copies its DNA before cell division. It occurs during the S-phase of the cell cycle, inside the nucleus in eukaryotic cells. The process is semi-conservative, meaning each daughter strand retains one original strand. Replication begins at multiple origins of replication. Helicase unwinds the DNA, while topoisomerase relieves torsional strain. Primase lays down RNA primers to initiate synthesis. DNA Polymerase III extends new strands in the 5' to 3' direction. DNA is synthesized continuously on the leading strand and in short segments (Okazaki fragments) on the lagging strand. DNA Polymerase I replaces RNA primers with DNA, and DNA Ligase seals the fragments. The strands are antiparallel, and the replication fork progresses bidirectionally."

Student Response: "Dna replication happens in nuclease, semi conservative, DNA polymerase does it, topoisomerase relieves tension while helices builds. can have many origins of replication, Okazaki fragments occur on leading strand, that's all I remember."

EXAMPLE OUTPUT:
{{
  "mentioned": [
    "Nucleus (as 'nuclease', likely typo)",
    "Semi-conservative replication",
    "DNA Polymerase",
    "Topoisomerase",
    "Helicase",
    "Origins of replication"
  ],
  "partial_mentions": [
    "Okazaki fragments (mentioned, but incorrect strand)",
    "Replication directionality (implied but not stated)"
  ],
  "missed": [
    "Primase",
    "RNA primers",
    "DNA Ligase",
    "Lagging strand",
    "Leading strand (technically misused)",
    "S-phase",
    "5' to 3' direction",
    "Antiparallel strands",
    "Replication fork",
    "DNA Polymerase I"
  ],
  "mentioned_count": 6,
  "total_key_concepts": 16,
  "score_fraction": "6/16",
  "feedback": "Nice recall of major enzymes like DNA polymerase, helicase, and topoisomerase! You also remembered that replication is semi-conservative and starts at multiple origins. Be careful though — Okazaki fragments occur on the *lagging* strand, not the leading strand. Try reviewing the sequence of events and strand directionality to deepen your understanding."
}}

Return a JSON object with this exact structure for the current student response. Be thorough in identifying all key concepts from the teaching content. Count partial mentions separately from full mentions. Provide encouraging but specific feedback that acknowledges what they got right and points out key areas to review."""

# ------------------- OPTIMIZED OPENAI CALLS -------------------
async def call_openai_async(messages: List[dict], max_tokens: int = 500, temperature: float = 0.3) -> str:
    """Async OpenAI call with optimized parameters"""
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            executor,
            lambda: openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=20,
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1    # Encourage variety
            )
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI call failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# ------------------- EXERCISE GENERATION ENDPOINT -------------------
@router.post("/blurting/exercises", response_model=BlurtingExerciseResponse)
async def generate_blurting_exercises(request: Request, data: BlurtingExerciseRequest):
    """Generate blurting exercises from teaching content"""
    try:
        user_id = extract_user_id(request, data)
        
        # Get context asynchronously
        context_result = await get_cached_context(user_id)
        context = context_result.get("context", {})

        # Generate improved prompt
        prompt = generate_exercise_prompt(data.teaching_block, context)

        # Async OpenAI call with optimal parameters
        messages = [
            {"role": "system", "content": "You are an expert educational content creator specializing in memory retrieval techniques and blurting exercises."},
            {"role": "user", "content": prompt}
        ]
        
        start_time = time.time()
        content = await call_openai_async(messages, max_tokens=600, temperature=0.4)
        print(f"⏱️ Exercise generation: {time.time() - start_time:.2f}s")

        # Parse and validate
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            cleaned = content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

        if not all(key in parsed for key in ["exercise_1", "exercise_2", "exercise_3", "key_concepts"]):
            raise ValueError("Invalid exercise structure")

        return BlurtingExerciseResponse(
            exercise_1=parsed["exercise_1"],
            exercise_2=parsed["exercise_2"],
            exercise_3=parsed["exercise_3"],
            key_concepts=parsed.get("key_concepts", [])
        )
        
    except Exception as e:
        print(f"❌ Exercise generation error: {e}")
        raise HTTPException(status_code=500, detail="Exercise generation failed")

# ------------------- FEEDBACK EVALUATION ENDPOINT -------------------
@router.post("/blurting/feedback", response_model=BlurtingFeedbackResponse)
async def evaluate_blurting_feedback(request: Request, data: BlurtingFeedbackRequest):
    """Evaluate blurting response with structured feedback"""
    try:
        user_id = extract_user_id(request, data)
        
        # Fetch context asynchronously for additional context
        context_result = await get_cached_context(user_id)
        context_prompt = None
        if context_result["context"]:
            weak_areas = context_result["context"].get("weak_areas", [])
            context_prompt = f"Focus on these concepts: {', '.join(weak_areas[:3])}" if weak_areas else None

        # Generate evaluation prompt
        prompt = generate_evaluation_prompt(
            data.teaching_content,
            data.blurted_response,
            context_prompt
        )

        # Async OpenAI call
        start_time = time.time()
        messages = [
            {"role": "system", "content": "You are an expert educational assessor specializing in memory recall evaluation and structured feedback."},
            {"role": "user", "content": prompt}
        ]
        content = await call_openai_async(messages, max_tokens=600, temperature=0.2)
        print(f"⏱️ OpenAI feedback call: {time.time() - start_time:.2f}s")

        # Parse response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Fallback parsing for malformed JSON
            print("⚠️ JSON parsing failed, attempting cleanup")
            cleaned = content.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)

        # Validate response structure
        required_keys = ["mentioned", "partial_mentions", "missed", "mentioned_count", "total_key_concepts", "score_fraction", "feedback"]
        if not all(key in parsed for key in required_keys):
            raise ValueError(f"Invalid response structure. Missing keys: {[k for k in required_keys if k not in parsed]}")

        # Fire-and-forget context logging
        asyncio.create_task(post_learning_event_async(
            user_id,
            data.teaching_content,
            parsed["missed"],
            parsed["feedback"],
            parsed["score_fraction"]
        ))

        return BlurtingFeedbackResponse(
            mentioned=parsed["mentioned"],
            partial_mentions=parsed["partial_mentions"], 
            missed=parsed["missed"],
            mentioned_count=parsed["mentioned_count"],
            total_key_concepts=parsed["total_key_concepts"],
            score_fraction=parsed["score_fraction"],
            feedback=parsed["feedback"]
        )

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        print(f"❌ Blurting feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback evaluation service error")
