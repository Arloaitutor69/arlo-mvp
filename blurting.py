from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List
from openai import OpenAI
import os
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

# ------------------- CONTEXT CACHE -------------------
context_cache: dict = {}
context_ttl = timedelta(minutes=5)

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

class ExerciseItem(BaseModel):
    prompt: str
    focus: str

class BlurtingExerciseResponse(BaseModel):
    exercise_1: ExerciseItem
    exercise_2: ExerciseItem

class BlurtingFeedbackRequest(BaseModel):
    exercise_question: str
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

# ------------------- JSON SCHEMAS FOR STRUCTURED OUTPUTS -------------------
BLURTING_EXERCISES_SCHEMA = {
    "name": "blurting_exercises_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "exercise_1": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "minLength": 10},
                    "focus": {"type": "string", "minLength": 10}
                },
                "required": ["prompt", "focus"],
                "additionalProperties": False
            },
            "exercise_2": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "minLength": 10},
                    "focus": {"type": "string", "minLength": 10}
                },
                "required": ["prompt", "focus"],
                "additionalProperties": False
            }
        },
        "required": ["exercise_1", "exercise_2"],
        "additionalProperties": False
    }
}

BLURTING_FEEDBACK_SCHEMA = {
    "name": "blurting_feedback_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "mentioned": {
                "type": "array",
                "items": {"type": "string", "minLength": 1}
            },
            "partial_mentions": {
                "type": "array",
                "items": {"type": "string", "minLength": 1}
            },
            "missed": {
                "type": "array",
                "items": {"type": "string", "minLength": 1}
            },
            "mentioned_count": {"type": "integer", "minimum": 0},
            "total_key_concepts": {"type": "integer", "minimum": 1},
            "score_fraction": {"type": "string", "pattern": "^\\d+/\\d+$"},
            "feedback": {"type": "string", "minLength": 20}
        },
        "required": ["mentioned", "partial_mentions", "missed", "mentioned_count", "total_key_concepts", "score_fraction", "feedback"],
        "additionalProperties": False
    }
}

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

# ------------------- CONTEXT LOGGING -------------------
async def post_learning_event_async(user_id: str, exercise_question: str, missed_concepts: List[str], feedback: str, score_fraction: str):
    """Async context posting with timeout protection"""
    topic = exercise_question.split('.')[0][:100] if exercise_question else "Blurting Exercise"
    
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

# ------------------- SYSTEM PROMPTS -------------------
EXERCISE_SYSTEM_PROMPT = """You are an expert learning scientist designing optimal blurting exercises based on teaching content.

BLURTING TECHNIQUE: Students recall information from memory without looking at materials. Most effective for:
- Factual information (dates, names, definitions)
- Sequential processes (steps, procedures)
- Lists and categorizations
- Detailed explanations of concepts

EXAMPLE INPUT:
Teaching Block: "DNA replication is the process by which a cell copies its DNA before cell division. It occurs during the S-phase of the cell cycle, inside the nucleus in eukaryotic cells. The process is semi-conservative, meaning each daughter strand retains one original strand. Replication begins at multiple origins of replication. Helicase unwinds the DNA, while topoisomerase relieves torsional strain. Primase lays down RNA primers to initiate synthesis. DNA Polymerase III extends new strands in the 5' to 3' direction. DNA is synthesized continuously on the leading strand and in short segments (Okazaki fragments) on the lagging strand. DNA Polymerase I replaces RNA primers with DNA, and DNA Ligase seals the fragments."

EXAMPLE OUTPUT STRUCTURE:
exercise_1:
  prompt: "List all the enzymes involved in DNA replication and describe what each one does."
  focus: "Factual recall of specific proteins and their functions"
exercise_2:
  prompt: "Describe the step-by-step process of DNA replication from initiation to completion."
  focus: "Sequential process recall and chronological understanding"

Create 2 distinct exercises targeting different memory retrieval patterns:
EXERCISE 1: Focus on detailed recall (facts, definitions, specific examples)
EXERCISE 2: Focus on process/sequence recall (steps, cause-effect, chronology)  

Make prompts specific and actionable."""

FEEDBACK_SYSTEM_PROMPT = """You are an expert educational assessor evaluating a student's blurting exercise response.

EVALUATION PROCESS:
1. Analyze the exercise question to identify ALL key concepts that should be included
2. Compare the student's response against these key concepts
3. Categorize performance into: MENTIONED, PARTIAL MENTIONS, MISSED

EXAMPLE INPUT:
Exercise Question: "List all the enzymes involved in DNA replication and describe what each one does."

Key concepts for this question:
- Helicase (unwinds DNA)
- Topoisomerase (relieves torsional strain)
- Primase (synthesizes RNA primers)
- DNA Polymerase III (extends new strands)
- DNA Polymerase I (replaces RNA primers with DNA)
- DNA Ligase (seals DNA fragments)

Student Response: "DNA polymerase extends the new strand, helicase unwinds DNA, topoisomerase relieves tension. There's also primase that makes primers. I think ligase does something too but I'm not sure what."

EXAMPLE OUTPUT STRUCTURE:
mentioned: [
  "DNA polymerase - extends new strand",
  "Helicase - unwinds DNA", 
  "Topoisomerase - relieves tension",
  "Primase - makes primers"
]
partial_mentions: [
  "DNA Ligase (mentioned but function unclear)"
]
missed: [
  "DNA Polymerase I vs III distinction",
  "Ligase specific function (seals DNA fragments)",
  "RNA primer specification"
]
mentioned_count: 4
total_key_concepts: 6
score_fraction: "4/6"
feedback: "Great job recalling the major enzymes! You correctly identified DNA polymerase, helicase, topoisomerase, and primase with their basic functions. You mentioned ligase but weren't sure about its function - it seals DNA fragments together. Try to be more specific about DNA polymerase types (I vs III) and remember that primers are specifically RNA primers."

Extract key concepts ONLY from what the specific exercise question asks for. Give encouraging feedback that acknowledges what they got right."""

# ------------------- ENDPOINTS -------------------
@router.post("/blurting/exercises", response_model=BlurtingExerciseResponse)
async def generate_blurting_exercises(request: Request, data: BlurtingExerciseRequest):
    """Generate blurting exercises from teaching content"""
    try:
        user_id = extract_user_id(request, data)
        
        # Get context
        context_result = await get_cached_context(user_id)
        context = context_result.get("context", {})
        
        # Build user prompt
        weak_areas = context.get("weak_areas", [])
        weak_areas_text = f"\n\nSTUDENT'S WEAK AREAS (prioritize these): {', '.join(weak_areas)}" if weak_areas else ""
        
        user_prompt = f"""TEACHING CONTENT:
{data.teaching_block[:1200]}
{weak_areas_text}

Create 2 distinct blurting exercises targeting different memory retrieval patterns."""

        # Prepare messages
        messages = [
            {"role": "system", "content": EXERCISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": BLURTING_EXERCISES_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)

        # Convert to Pydantic models
        return BlurtingExerciseResponse(
            exercise_1=ExerciseItem(
                prompt=parsed_data["exercise_1"]["prompt"],
                focus=parsed_data["exercise_1"]["focus"]
            ),
            exercise_2=ExerciseItem(
                prompt=parsed_data["exercise_2"]["prompt"],
                focus=parsed_data["exercise_2"]["focus"]
            )
        )
        
    except Exception as e:
        print(f"❌ Exercise generation error: {e}")
        raise HTTPException(status_code=500, detail="Exercise generation failed")

@router.post("/blurting/feedback", response_model=BlurtingFeedbackResponse)
async def evaluate_blurting_feedback(request: Request, data: BlurtingFeedbackRequest):
    """Evaluate blurting response against the exercise question"""
    try:
        user_id = extract_user_id(request, data)
        
        # Get context
        context_result = await get_cached_context(user_id)
        context_prompt = None
        if context_result["context"]:
            weak_areas = context_result["context"].get("weak_areas", [])
            context_prompt = f"Focus on these concepts: {', '.join(weak_areas[:3])}" if weak_areas else None

        # Build user prompt
        context_section = f"\n\nStudent's learning context (weak areas to focus on):\n{context_prompt}" if context_prompt else ""
        
        user_prompt = f"""EXERCISE QUESTION (what the student was asked to recall):
{data.exercise_question}

STUDENT'S BLURTED RESPONSE:
{data.blurted_response[:800]}
{context_section}

Evaluate the student's performance by identifying key concepts from the question and categorizing their recall."""

        # Prepare messages
        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": BLURTING_FEEDBACK_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)

        # Fire-and-forget context logging
        asyncio.create_task(post_learning_event_async(
            user_id,
            data.exercise_question,
            parsed_data["missed"],
            parsed_data["feedback"],
            parsed_data["score_fraction"]
        ))

        return BlurtingFeedbackResponse(
            mentioned=parsed_data["mentioned"],
            partial_mentions=parsed_data["partial_mentions"], 
            missed=parsed_data["missed"],
            mentioned_count=parsed_data["mentioned_count"],
            total_key_concepts=parsed_data["total_key_concepts"],
            score_fraction=parsed_data["score_fraction"],
            feedback=parsed_data["feedback"]
        )

    except Exception as e:
        print(f"❌ Blurting feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback evaluation service error")
