import os
import json
import uuid
import asyncio
import hashlib
from typing import List, Optional, Dict, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from openai import OpenAI
import httpx
from functools import lru_cache

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
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
    technique: str  # Primary technique for backward compatibility
    techniques: List[str]  # New field for multiple techniques
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

# --- JSON Schema for structured outputs --- #
STUDY_PLAN_SCHEMA = {
    "name": "study_plan_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "units_to_cover": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "minLength": 1
                }
            },
            "pomodoro": {
                "type": "string",
                "enum": ["25/5", "30/5", "45/15", "50/10"]
            },
            "techniques": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "string",
                    "enum": ["flashcards", "feynman", "quiz", "blurting"]
                }
            },
            "blocks": {
                "type": "array",
                "minItems": 2,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": {
                        "unit": {
                            "type": "string",
                            "minLength": 1
                        },
                        "techniques": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {
                                "type": "string",
                                "enum": ["flashcards", "feynman", "quiz", "blurting"]
                            }
                        },
                        "description": {
                            "type": "string",
                            "minLength": 100
                        },
                        "duration": {
                            "type": "integer",
                            "minimum": 10,
                            "maximum": 30
                        }
                    },
                    "required": ["unit", "techniques", "description", "duration"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["units_to_cover", "pomodoro", "techniques", "blocks"],
        "additionalProperties": False
    }
}

def create_fallback_study_plan(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> dict:
    """Create a fallback study plan when JSON parsing fails"""
    num_blocks, block_duration = calculate_optimal_blocks(duration)
    topic = objective or "Study Session"
    
    # Create basic units based on available content
    if parsed_summary and len(parsed_summary) > 100:
        units = [f"Section {i+1}" for i in range(min(num_blocks, 4))]
    elif objective:
        units = [f"{topic} - Part {i+1}" for i in range(min(num_blocks, 3))]
    else:
        units = ["Review Session", "Practice Problems", "Key Concepts"]
    
    # Pad units to match number of blocks if needed
    while len(units) < num_blocks:
        units.append(f"Additional Study {len(units) + 1}")
    
    blocks = []
    for i in range(num_blocks):
        unit_name = units[i] if i < len(units) else f"Study Block {i+1}"
        
        blocks.append({
            "unit": unit_name,
            "techniques": ["feynman", "quiz"] if i % 2 == 0 else ["flashcards", "blurting"],
            "description": f"Comprehensive study of {unit_name}. Focus on key concepts, important details, and practical applications. Use active recall and spaced repetition techniques to reinforce learning.",
            "duration": block_duration
        })
    
    return {
        "units_to_cover": units[:num_blocks],
        "pomodoro": "25/5",
        "techniques": ["feynman", "flashcards", "quiz", "blurting"],
        "blocks": blocks
    }

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

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt with better structure and examples"""
    
    num_blocks, block_duration = calculate_optimal_blocks(duration)
    
    # Build content section with better truncation
    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"
    
    if parsed_summary:
        # Truncate more aggressively to prevent token overflow
        truncated_summary = parsed_summary[:2500]  # Reduced from 4500
        content_section += f"SOURCE MATERIAL TO COVER:\n{truncated_summary}\n\n"
    
    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    prompt = f"""You are an expert curriculum designer creating a study plan.

{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} learning blocks  
- Each block should be {block_duration} minutes long

AVAILABLE TECHNIQUES (choose 1-3 per block based on what's best for each unit):
• flashcards: Spaced repetition for memorization
• feynman: Explain concepts in simple terms
• quiz: Active recall testing
• blurting: Free recall without prompts

REQUIREMENTS:
- Each block needs a clear unit/topic name
- Choose 1-3 BEST techniques for each specific unit/topic - more is better to reinforce with active recall 
- You can use any combination or sequence of techniques within a block
- Focus on what will help the student learn THIS specific content most effectively
- Each block must cover distinct, non redundant and non-overlapping content that builds progressively toward complete mastery of the subject

CONTENT REQUIREMENTS FOR EACH BLOCK:
1. Each description must be a complete self contained lesson including:
2. Key definitions and examples, Important formulas, equations, or principles, Specific facts, data points, or details to remember
3. focus on most important details as oppose to minor tangential. Cater to highschool/undergraduate college info unless otherwise specified
4. Each description should be 100-200 words
5. the collection of study topics and descriptions should cover the entirety of the content the student wants to learn in that session

EXAMPLE QUALITY DESCRIPTION:
For an input topic  "Cellular Biology" break down into Cell Structure and Function", "Cellular Respiration","Photosynthesis", "Cell Communication and Signaling","Cell Cycle and Division"
Example Photosynthesis Study Plan Description: "Understanding the Basics of Photosynthesis: 1. Master Equation – 6CO₂ + 6H₂O + light → C₆H₁₂O₆ + 6O₂. 2. Autotrophs (contrast with heterotrophs during lesson). 3. Chloroplasts. 4. Thylakoids – site of light-dependent reactions. 5. Stroma – site of light-independent reactions (connect to Calvin cycle in teaching flow). 6. Pigments – chlorophyll a & b, accessory pigments. 7. Gas Exchange (highlight water loss trade-off). 8. Carbon Fixation Pathways – C3, C4, CAM plants. 9. Photorespiration – RuBisCO fixing O₂ instead of CO₂ (stress why it reduces efficiency). 10. Light-Dependent Reactions. 11. Light-Independent Reactions. 12. Importance – provides O₂ and organic molecules for life (wrap up with global impact facts and relevance)."
For an input topic of "AP US History 1900-present", break down into Progressive Era and Early 20th Century Reform", "The Great Depression and New Deal", "World War II and the Home Front", "Cold War Era Politics and Society", "21st Century America: Globalization and Contemporary Issues"
Example description for Cold War Era Politics and Society: 1. Defining the Cold War – ideological struggle between democracy/capitalism and communism. 2. Superpowers – U.S. vs USSR rivalry. 3. Containment Policy (Truman Doctrine, Marshall Plan, NATO) 4. Nuclear Arms Race 5. Space Race – Sputnik, NASA, Moon landing. 6. Proxy Wars – Korea, Vietnam, Afghanistan (unpack impacts of each breifly). 7. Domestic Impact – Red Scare, McCarthyism, civil defense drills. 8. Berlin Crisis – Berlin Airlift, Berlin Wall. 9. Cuban Missile Crisis 10. Détente – easing tensions, SALT treaties. 11. Reagan Era – military buildup, 'Evil Empire' speech, Strategic Defense Initiative. 12. End of the Cold War (fall of Berlin Wall, collapse of Soviet Union in 1991)"


Create a study plan with exactly {num_blocks} blocks of {block_duration} minutes each."""
    
    return prompt

async def update_context_async(payload: dict) -> bool:
    """Asynchronously update context API with better error handling"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:  # Reduced timeout
            response = await client.post(f"{CONTEXT_API}/api/context/update", json=payload)
            return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Context update failed: {e}")
        return False

def generate_gpt_plan(prompt: str, objective: Optional[str] = None, parsed_summary: Optional[str] = None, duration: int = 60) -> dict:
    """Generate study plan with GPT-5-nano structured outputs"""
    
    print(f"📏 Prompt length: {len(prompt)} characters")
    
    try:
        print(f"🤖 Calling GPT-5-nano...")
        
        # Use GPT-5-nano structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are an expert curriculum designer. Create comprehensive study plans with exactly the requested number of blocks. Each block must be educational, actionable, and contain substantial learning content with specific examples, definitions, and key facts."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": STUDY_PLAN_SCHEMA
            },
            reasoning_effort="low",
            max_output_tokens="4000"
        )

        raw_output = response.choices[0].message.content.strip()
        
        print(f"📝 Raw GPT response length: {len(raw_output)} chars")
        print(f"📝 First 300 chars: {raw_output[:300]}...")
        
        # Parse the guaranteed valid JSON response
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # Use fallback response
            parsed_output = create_fallback_study_plan(objective, parsed_summary, duration)
        
        blocks = parsed_output.get("blocks", [])
        if not blocks:
            print("❌ No blocks generated, using fallback")
            parsed_output = create_fallback_study_plan(objective, parsed_summary, duration)
            blocks = parsed_output.get("blocks", [])
        
        print(f"✅ GPT generated valid response with {len(blocks)} blocks")
        print(f"📊 Units: {len(parsed_output.get('units_to_cover', []))}")
        print(f"🔧 Techniques: {len(parsed_output.get('techniques', []))}")
        return parsed_output
        
    except Exception as e:
        print(f"🔥 GPT generation failed: {e}")
        print(f"Using fallback study plan...")
        return create_fallback_study_plan(objective, parsed_summary, duration)

# --- Main Endpoint ---
@router.post("/study-session", response_model=StudyPlanResponse)
async def generate_plan(data: StudyPlanRequest, request: Request):
    """Generate comprehensive study plan with enhanced logging and error handling"""
    
    print(f"🚀 Starting study plan generation...")
    print(f"📋 Request: duration={data.duration}, objective={bool(data.objective)}, summary={bool(data.parsed_summary)}")
    
    try:
        user_id = extract_user_id(request)
        print(f"👤 User ID: {user_id}")
        
        # Build enhanced prompt
        print("📝 Building enhanced prompt...")
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)
        
        # Generate plan with GPT
        print("🤖 Calling GPT...")
        parsed = generate_gpt_plan(prompt, data.objective, data.parsed_summary, data.duration)
        
        # Extract plan components
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")
        
        print(f"📦 Processing {len(blocks_json)} blocks...")
        
        # Build study blocks
        blocks = []
        context_tasks = []
        total_time = 0
        
        for idx, item in enumerate(blocks_json):
            try:
                unit = item.get("unit", f"Unit {idx + 1}")
                techniques_list = item.get("techniques", ["feynman"])
                primary_technique = techniques_list[0] if techniques_list else "feynman"
                description = item.get("description", "Study the assigned material")
                duration = item.get("duration", 12)
                block_id = f"block_{uuid.uuid4().hex[:8]}"
                
                # Create study block with both single technique (backward compatibility) and multiple techniques
                study_block = StudyBlock(
                    id=block_id,
                    unit=unit,
                    technique=primary_technique,  # Primary technique for backward compatibility
                    techniques=techniques_list,   # New field for multiple techniques
                    phase=primary_technique,
                    tool=primary_technique,
                    lovable_component="text-block",
                    duration=duration,
                    description=description,
                    position=idx
                )
                
                blocks.append(study_block)
                total_time += duration
                
                print(f"✅ Block {idx + 1}: {unit} - {techniques_list} ({duration}min)")
                print(f"   Description: {description[:100]}...")
                
                # Prepare context update (async) - use primary technique
                context_payload = {
                    "source": "session_planner",
                    "user_id": user_id,
                    "current_topic": f"{unit} — {primary_technique}",
                    "learning_event": {
                        "concept": unit,
                        "phase": primary_technique,
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Planned {' + '.join(techniques_list)} session: {description[:200]}...",
                        "repetition_count": 0,
                        "review_scheduled": False
                    }
                }
                
                context_tasks.append(update_context_async(context_payload))
                
            except Exception as e:
                print(f"❌ Error processing block {idx}: {e}")
                continue
        
        # Execute context updates with timeout protection
        print(f"📤 Sending {len(context_tasks)} context updates...")
        try:
            context_results = await asyncio.wait_for(
                asyncio.gather(*context_tasks, return_exceptions=True),
                timeout=10.0
            )
            successful_updates = sum(1 for result in context_results if result is True)
            print(f"✅ {successful_updates}/{len(context_tasks)} context updates successful")
        except asyncio.TimeoutError:
            print("⏰ Context updates timed out")
            successful_updates = 0
        
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
                await asyncio.wait_for(update_context_async(final_payload), timeout=5.0)
                print("🧠 Synthesis trigger sent")
            except Exception as e:
                print(f"⚠️ Synthesis trigger failed: {e}")
        
        print(f"🎉 Study plan generated successfully!")
        print(f"📊 Total: {len(blocks)} blocks, {total_time} minutes")
        
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
