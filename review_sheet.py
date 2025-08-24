# FIXED REVIEW SHEET MODULE - GPT-5-NANO STRUCTURED OUTPUTS

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from openai import OpenAI
import os
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CRITICAL FIX: Import context function directly instead of HTTP calls
from context import get_cached_context_fast  # ✅ correct

# ---------------------------
# Setup
# ---------------------------
app = FastAPI()
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for non-blocking operations
executor = ThreadPoolExecutor(max_workers=4)

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
    study_tips: List[str]

# --- JSON Schema for structured outputs --- #
REVIEW_SHEET_SCHEMA = {
    "name": "review_sheet_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "summary": {
                "type": "string",
                "minLength": 50,
                "maxLength": 300
            },
            "memorization_facts": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 10
                }
            },
            "weak_areas": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {
                    "type": "string",
                    "minLength": 5
                }
            },
            "major_topics": {
                "type": "array",
                "minItems": 3,
                "maxItems": 4,
                "items": {
                    "type": "string",
                    "minLength": 3
                }
            },
            "study_tips": {
                "type": "array",
                "minItems": 2,
                "maxItems": 3,
                "items": {
                    "type": "string",
                    "minLength": 15
                }
            }
        },
        "required": ["summary", "memorization_facts", "weak_areas", "major_topics", "study_tips"],
        "additionalProperties": False
    }
}

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
# Context Processing (UNCHANGED)
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
# Enhanced System and User Prompts
# ---------------------------
def build_system_prompt() -> str:
    return """You are Arlo, an expert learning coach specializing in memory consolidation and spaced repetition. Generate concise, actionable bedtime review sheets that optimize long-term retention.

QUALITY STANDARDS:
- Create personalized content based on the student's actual learning session
- Focus on information that benefits from sleep consolidation (facts, procedures, connections)
- Provide specific, actionable study recommendations
- Use encouraging but realistic language
- Prioritize definitions, formulas, and key concepts for memorization

CONTENT GUIDELINES:
- Summary: 2-3 sentences highlighting main achievements and progress
- Memorization facts: 3-5 specific facts perfect for bedtime review
- Major topics: 3-4 main subjects covered in the session
- Weak areas: 2-3 specific areas needing more practice with constructive framing
- Study tips: 2-3 personalized, actionable recommendations for tomorrow

EXAMPLE OUTPUT:

Summary: You made excellent progress today working through cellular respiration concepts and successfully tackled challenging ATP synthesis problems. Your understanding of the electron transport chain has improved significantly, and you demonstrated good problem-solving skills when analyzing metabolic pathways.

Memorization Facts:
- Cellular respiration produces approximately 32-38 ATP molecules per glucose molecule
- The electron transport chain occurs in the inner mitochondrial membrane
- NADH produces about 2.5 ATP molecules while FADH₂ produces about 1.5 ATP molecules
- Glycolysis occurs in the cytoplasm and does not require oxygen

Major Topics:
- Cellular respiration overview and stages
- ATP synthesis and energy production
- Electron transport chain mechanisms
- Metabolic pathway analysis

Weak Areas:
- Distinguishing between aerobic and anaerobic respiration pathways
- Calculating net ATP yield in different metabolic scenarios

Study Tips:
- Practice drawing the electron transport chain from memory tomorrow morning
- Use flashcards to drill the ATP yield numbers from different molecules
- Try explaining cellular respiration to someone else to reinforce your understanding"""

def build_user_prompt(processed_context: dict) -> str:
    """Build a focused, optimized prompt for better review generation"""
    
    recent_topics = processed_context.get("recent_topics", [])
    struggle_areas = processed_context.get("struggle_areas", [])
    key_facts = processed_context.get("key_facts", [])
    learning_goals = processed_context.get("learning_goals", [])
    
    return f"""Generate a personalized bedtime review sheet for optimal memory consolidation based on today's learning session.

STUDENT'S LEARNING SESSION DATA:
Topics covered: {', '.join(recent_topics[:5]) if recent_topics else 'General study session'}
Key facts learned: {', '.join(key_facts[:8]) if key_facts else 'Various concepts'}
Areas of difficulty: {', '.join(struggle_areas[:4]) if struggle_areas else 'None identified'}
Learning goals: {', '.join(learning_goals[:3]) if learning_goals else 'General mastery'}

Create a comprehensive review sheet with:
1. Summary: 2-3 sentences highlighting main achievements from today's session
2. Memorization facts: 3-5 specific facts perfect for bedtime review (prioritize definitions, formulas, key concepts)
3. Major topics: 3-4 main subjects covered during the study session
4. Weak areas: 2-3 specific areas needing more practice (be constructive and specific)
5. Study tips: 2-3 personalized, actionable recommendations for tomorrow's study session

Focus on content that benefits most from sleep consolidation and will help the student retain and apply what they learned."""

# ---------------------------
# Optimized GPT API Call with Structured Outputs
# ---------------------------
async def call_gpt_structured(processed_context: dict) -> dict:
    """Optimized GPT call with structured outputs for review generation"""
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(processed_context)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": REVIEW_SHEET_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        # This should never happen with structured outputs, but just in case
        logger.error(f"Failed to parse response as JSON: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        logger.error(f"GPT API call failed: {e}")
        raise HTTPException(status_code=500, detail="AI service temporarily unavailable")

# ---------------------------
# Main Endpoint (UPDATED WITH STRUCTURED OUTPUTS)
# ---------------------------
@router.post("/review-sheet", response_model=ReviewSheet)
async def generate_review_sheet(request: Request, data: ReviewRequest):
    """Generate optimized review sheet with direct context import - NO MORE TIMEOUTS"""
    
    try:
        user_id = extract_user_id(request, data)
        
        # CRITICAL FIX: Direct function call instead of HTTP request
        context_result = get_cached_context_fast(user_id)
        context = context_result.get("context", {})
        
        # Log context source for debugging
        logger.info(f"Using context from: {context_result.get('source')}, "
                   f"age: {context_result.get('age_minutes', 0)} min, "
                   f"user: {user_id}")
        
        # Process context for optimal review generation
        processed_context = process_context_for_review(context)
        
        logger.info(f"Generating review for user {user_id}")
        
        # Call GPT with structured outputs
        parsed_data = await call_gpt_structured(processed_context)
        
        # Convert to Pydantic model
        review_sheet = ReviewSheet(
            summary=parsed_data["summary"],
            memorization_facts=parsed_data["memorization_facts"],
            major_topics=parsed_data["major_topics"],
            weak_areas=parsed_data["weak_areas"],
            study_tips=parsed_data["study_tips"]
        )
        
        logger.info(f"Successfully generated review sheet for user {user_id}")
        return review_sheet
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating review sheet: {e}")
        # Return basic fallback - never fail completely
        return ReviewSheet(
            summary="You've completed a productive study session today. Great work staying committed to your learning goals and making consistent progress in your studies.",
            memorization_facts=[
                "Consistent daily review improves long-term retention by 60%",
                "Sleep consolidation helps transfer information from short-term to long-term memory",
                "Active recall techniques are more effective than passive re-reading"
            ],
            major_topics=["General study session completed", "Learning strategies reviewed", "Knowledge consolidation"],
            weak_areas=["Consider tracking specific topics for more detailed feedback", "Focus on active practice techniques"],
            study_tips=[
                "Review these facts again tomorrow morning for better retention",
                "Focus on active recall techniques in your next session",
                "Try explaining concepts out loud to reinforce understanding"
            ]
        )

# ---------------------------
# Health Check Endpoint
# ---------------------------
@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# ---------------------------
# Attach router
# ---------------------------
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10001, log_level="info")
