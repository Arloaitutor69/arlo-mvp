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

def validate_block_content(block: dict) -> bool:
    """Validate that block has substantial, educational content"""
    description = block.get("description", "")
    
    # Minimum length check
    if len(description) < 150:
        return False
    
    # Check for educational markers
    educational_markers = [
        "definition", "process", "example", "step", "key", "important", 
        "remember", "concept", "principle", "formula", "equation", "theory"
    ]
    
    description_lower = description.lower()
    marker_count = sum(1 for marker in educational_markers if marker in description_lower)
    
    # Should have at least 2 educational markers
    return marker_count >= 2

def get_technique_distribution(num_blocks: int) -> List[str]:
    """Get optimal technique distribution based on number of blocks"""
    # Base techniques for different block counts
    technique_pools = {
        2: ["arlo_teaching", "quiz"],
        3: ["arlo_teaching", "flashcards", "quiz"],
        4: ["arlo_teaching", "flashcards", "feynman", "quiz"],
        5: ["arlo_teaching", "flashcards", "feynman", "quiz", "blurting"],
        6: ["arlo_teaching", "flashcards", "feynman", "quiz", "blurting", "flashcards"],
        7: ["arlo_teaching", "flashcards", "feynman", "quiz", "blurting", "flashcards", "quiz"],
        8: ["arlo_teaching", "flashcards", "feynman", "quiz", "blurting", "flashcards", "quiz", "feynman"]
    }
    
    return technique_pools.get(num_blocks, ["arlo_teaching", "flashcards", "quiz", "feynman"])

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

    prompt = f"""You are an expert curriculum designer with deep knowledge of cognitive science and learning optimization.

{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} learning blocks  
- Each block should be {block_duration}-15 minutes long

SCIENTIFICALLY-OPTIMIZED TECHNIQUE SELECTION:
You must analyze the content and choose techniques based on cognitive science research:

CONTENT ANALYSIS RULES:
1. FACTUAL CONTENT (definitions, dates, formulas, vocabulary):
   - High information density, requires memorization
   - Use: flashcards → quiz → blurting (spacing effect + testing effect)

2. CONCEPTUAL CONTENT (theories, principles, cause-effect relationships):
   - Requires deep understanding and connections
   - Use: arlo_teaching → feynman → quiz (elaborative interrogation + generation effect)

3. PROCEDURAL CONTENT (step-by-step processes, problem-solving):
   - Requires practice and application
   - Use: arlo_teaching → blurting → quiz → feynman (practice effect + reflection)

4. MIXED CONTENT:
   - Combine techniques strategically
   - Always start with arlo_teaching for initial encoding
   - End with quiz for retrieval practice (testing effect)

TECHNIQUE DESCRIPTIONS (Based on Learning Science):
• arlo_teaching: Initial encoding and guided instruction (reduces cognitive load)
• flashcards: Spaced repetition for factual recall (spacing effect, testing effect)
• feynman: Elaborative explanation in simple terms (generation effect, elaborative interrogation)
• quiz: Active retrieval practice (testing effect, desirable difficulties)
• blurting: Free recall without cues (retrieval practice, transfer appropriate processing)

SELECTION STRATEGY:
1. Analyze each unit's content type (factual/conceptual/procedural)
2. Choose techniques that match the cognitive demands
3. Sequence techniques to build from encoding → practice → retrieval
4. Ensure variety to maintain engagement (attention restoration theory)

CONTENT REQUIREMENTS FOR EACH BLOCK:
Each description must be a complete mini-lesson including:
1. Key definitions with clear examples
2. Step-by-step processes or procedures
3. Important formulas, equations, or principles
4. Common misconceptions students should avoid
5. Real-world applications or examples
6. Specific facts, data points, or details to remember

QUALITY STANDARDS:
- Each description should be 200-400 words
- Include concrete examples, not just abstract concepts
- Provide actionable learning content, not just topic overviews
- Make content self-contained (other modules only see the description)

EXAMPLE OUTPUT FORMAT:
{{
  "content_analysis": {{
    "primary_content_type": "mixed",
    "reasoning": "Contains both factual information (equations, definitions) and conceptual understanding (processes, principles)",
    "technique_rationale": "Starting with teaching for encoding, using flashcards for factual recall, Feynman for conceptual understanding, quiz for testing effect"
  }},
  "units_to_cover": ["Photosynthesis Overview", "Light Reactions", "Calvin Cycle"],
  "pomodoro": "25/5",
  "techniques": ["arlo_teaching", "flashcards", "feynman", "quiz"],
  "blocks": [
    {{
      "unit": "Photosynthesis Overview",
      "technique": "arlo_teaching",
      "content_type": "conceptual",
      "description": "Photosynthesis is the fundamental process where plants convert light energy into chemical energy stored as glucose. Key equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2 + ATP. This process occurs in chloroplasts, specifically in two stages: light reactions (thylakoids) and Calvin cycle (stroma). Important principle: Plants are autotrophs, meaning they produce their own food. Common misconception: plants don't need oxygen - actually, they produce oxygen during photosynthesis but consume it during cellular respiration at night. Real-world significance: photosynthesis produces approximately 70% of Earth's oxygen and forms the foundation of most food chains. Key terms to remember: chlorophyll (green pigment that captures light), stomata (pores for gas exchange), and ATP (energy currency). Process efficiency: only about 1-2% of sunlight is converted to chemical energy.",
      "duration": 12
    }}
  ]
}}

CRITICAL: Analyze the content type for each unit and select techniques based on cognitive science, not just following a preset sequence. Justify your technique choices in the content_analysis section.

Return only valid JSON with no markdown formatting or additional text."""
    
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
    """Generate study plan with GPT with retries and validation"""
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🤖 GPT attempt {attempt + 1}/{max_retries + 1}")
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert curriculum designer. Return only valid JSON with comprehensive educational content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused content
                max_tokens=2500,  # Ensure complete responses
                top_p=0.9
            )

            raw_response = completion.choices[0].message.content.strip()
            
            # Clean up response
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:-3].strip()
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:-3].strip()
            
            # Parse JSON
            parsed = json.loads(raw_response)
            
            # Validate required fields
            if not all(key in parsed for key in ["blocks", "units_to_cover", "techniques"]):
                raise ValueError("Missing required fields in GPT response")
            
            blocks = parsed.get("blocks", [])
            if not blocks:
                raise ValueError("No blocks generated")
            
            # Validate block content quality
            valid_blocks = [block for block in blocks if validate_block_content(block)]
            
            if len(valid_blocks) < len(blocks) * 0.7:  # At least 70% should be valid
                raise ValueError(f"Only {len(valid_blocks)}/{len(blocks)} blocks passed validation")
            
            parsed["blocks"] = valid_blocks
            print(f"✅ GPT generated {len(valid_blocks)} valid blocks")
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
    """Generate comprehensive study plan with improved content quality and performance"""
    
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
        
        # Build study blocks
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
