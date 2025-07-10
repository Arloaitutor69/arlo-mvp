import openai
import os
import json
import uuid
import asyncio
import hashlib
from typing import List, Optional, Dict, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import httpx
from functools import lru_cache

# Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# --- Models ---
class StudyPlanRequest(BaseModel):
    objective: Optional[str] = None  # Freeform input from student
    parsed_summary: Optional[str] = None  # Optional PDF parser output
    duration: int = 60

class StudyBlock(BaseModel):
    id: str
    unit: str
    technique: str
    phase: str
    tool: str
    lovable_component: str
    duration: int
    description: str
    position: int
    custom: bool = False
    user_notes: Optional[str] = None
    payload: Optional[Dict] = None

class StudyPlanResponse(BaseModel):
    session_id: str
    topic: str
    total_duration: int
    pomodoro: str
    units_to_cover: List[str]
    techniques: List[str]
    blocks: List[StudyBlock]

# --- Utility Functions ---
def extract_user_id(request: Request) -> str:
    """Extract user ID from request state or headers"""
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        return user_info["sub"]
    elif request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

def calculate_optimal_blocks(duration: int) -> Tuple[int, int]:
    """Calculate optimal number of blocks and duration per block"""
    if duration <= 30:
        return 2, 12
    elif duration <= 60:
        return 4, 12
    elif duration <= 90:
        return 6, 13
    else:
        return 8, 15

def create_content_hash(objective: str, parsed_summary: str, duration: int) -> str:
    """Create hash for caching purposes"""
    content = f"{objective or ''}{parsed_summary or ''}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

# Removed get_technique_distribution - let GPT choose techniques freely

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt with better structure and examples"""
    
    num_blocks, block_duration = calculate_optimal_blocks(duration)
    
    # Build content section
    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"
    
    if parsed_summary:
        # Use more content but still reasonable for context
        content_section += f"SOURCE MATERIAL TO COVER:\n{parsed_summary[:4500]}\n\n"
    
    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    prompt = f"""You are an expert curriculum designer creating a study plan.

{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} learning blocks  
- Each block should be {block_duration} minutes long

AVAILABLE TECHNIQUES (choose what's best for each unit):
• arlo_teaching: Interactive teaching and explanation
• flashcards: Spaced repetition for memorization
• feynman: Explain concepts in simple terms
• quiz: Active recall testing
• blurting: Free recall without prompts

REQUIREMENTS:
- Each block needs a clear unit/topic name
- Choose the BEST technique for each specific unit/topic
- You can use the same technique multiple times if optimal
- You can use any combination or sequence of techniques
- Focus on what will help the student learn THIS specific content most effectively
- Provide helpful description for each block
- Make descriptions practical and actionable

Return ONLY valid JSON in this exact format:
{{
  "units_to_cover": ["Unit 1", "Unit 2", "Unit 3"],
  "pomodoro": "25/5",
  "techniques": ["arlo_teaching", "flashcards", "quiz", "feynman"],
  "blocks": [
    {{
      "unit": "Unit Name",
      "technique": "arlo_teaching",
      "description": "Clear description of what to study and how",
      "duration": {block_duration}
    }}
  ]
}}

Make sure to include exactly {num_blocks} blocks in your response."""
    
    return prompt

async def update_context_async(payload: dict) -> bool:
    """Asynchronously update context API"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{CONTEXT_API}/api/context/update", json=payload)
            return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Context update failed: {e}")
        return False

def generate_gpt_plan(prompt: str, max_retries: int = 2) -> dict:
    """Generate study plan with GPT with retries - NO VALIDATION"""
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🤖 GPT attempt {attempt + 1}/{max_retries + 1}")
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert curriculum designer. Return only valid JSON with study plan blocks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
                top_p=0.9
            )

            raw_response = completion.choices[0].message.content.strip()
            print(f"📝 Raw GPT response: {raw_response[:200]}...")
            
            # Clean up response
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:-3].strip()
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:-3].strip()
            
            # Parse JSON
            parsed = json.loads(raw_response)
            
            # Only validate basic structure - no content validation
            if not all(key in parsed for key in ["blocks", "units_to_cover", "techniques"]):
                raise ValueError("Missing required fields in GPT response")
            
            blocks = parsed.get("blocks", [])
            if not blocks:
                raise ValueError("No blocks generated")
            
            print(f"✅ GPT generated {len(blocks)} blocks - accepting all")
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"🔥 JSON parsing failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail="Failed to parse GPT response as JSON")
                
        except Exception as e:
            print(f"🔥 GPT generation failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Max retries exceeded")

# --- Main Endpoint ---
@router.post("/study-session", response_model=StudyPlanResponse)
async def generate_plan(data: StudyPlanRequest, request: Request):
    """Generate comprehensive study plan - simplified version"""
    
    try:
        user_id = extract_user_id(request)
        
        # Build enhanced prompt
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)
        
        # Generate plan with GPT
        parsed = generate_gpt_plan(prompt)
        
        # Extract plan components
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")
        
        # Build study blocks - accept whatever GPT gives us
        blocks = []
        context_tasks = []
        total_time = 0
        
        for idx, item in enumerate(blocks_json):
            unit = item.get("unit", f"Unit {idx + 1}")
            technique = item.get("technique", "feynman")
            description = item.get("description", "Study the assigned material")
            duration = item.get("duration", 12)
            block_id = f"block_{uuid.uuid4().hex[:8]}"
            
            # Create study block
            study_block = StudyBlock(
                id=block_id,
                unit=unit,
                technique=technique,
                phase=technique,
                tool=technique,
                lovable_component="text-block",
                duration=duration,
                description=description,
                position=idx
            )
            
            blocks.append(study_block)
            total_time += duration
            
            print(f"📋 Block {idx + 1}: {unit} - {technique} ({duration}min)")
            print(f"   Description: {description[:100]}...")
            
            # Prepare context update (async)
            context_payload = {
                "source": "session_planner",
                "user_id": user_id,
                "current_topic": f"{unit} — {technique}",
                "learning_event": {
                    "concept": unit,
                    "phase": technique,
                    "confidence": None,
                    "depth": None,
                    "source_summary": f"Planned {technique} session: {description[:200]}...",
                    "repetition_count": 0,
                    "review_scheduled": False
                }
            }
            
            context_tasks.append(update_context_async(context_payload))
        
        # Execute all context updates concurrently
        print(f"📤 Sending {len(context_tasks)} context updates...")
        context_results = await asyncio.gather(*context_tasks, return_exceptions=True)
        successful_updates = sum(1 for result in context_results if result is True)
        print(f"✅ {successful_updates}/{len(context_tasks)} context updates successful")
        
        # Send final synthesis trigger if any updates succeeded
        if successful_updates > 0:
            try:
                final_payload = {
                    "source": "session_planner",
                    "user_id": user_id,
                    "current_topic": "Complete Study Plan",
                    "learning_event": {
                        "concept": data.objective or "Generated Study Plan",
                        "phase": "planning",
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Comprehensive study plan with {len(blocks)} blocks covering: {', '.join(units[:3])}{'...' if len(units) > 3 else ''}",
                        "repetition_count": 0,
                        "review_scheduled": False
                    },
                    "trigger_synthesis": True
                }
                await update_context_async(final_payload)
                print("🧠 Synthesis trigger sent")
            except Exception as e:
                print(f"⚠️ Synthesis trigger failed: {e}")
        
        # Return complete study plan
        return StudyPlanResponse(
            session_id=session_id,
            topic=data.objective or "Study Plan from Uploaded Content",
            total_duration=total_time,
            pomodoro=pomodoro,
            units_to_cover=units,
            techniques=techniques,
            blocks=blocks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Study plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
