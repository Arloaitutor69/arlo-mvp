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
- Choose 1-3 BEST techniques for each specific unit/topic
- You can use any combination or sequence of techniques within a block
- Focus on what will help the student learn THIS specific content most effectively
- Each block must cover distinct, non redundant and non-overlapping content that builds progressively toward complete mastery of the subject

CONTENT REQUIREMENTS FOR EACH BLOCK:
Each description must be a complete self contained mini-lesson including:
1. Key definitions and examples
2. Important formulas, equations, or principles
3. Common misconceptions students should avoid
4. Specific facts, data points, or details to remember
5. MOST IMPORTANT: it should include EVERY SINGLE relevant sub topic to ensure student is fully prepared once they learned all those topics, and all subtopics should be relevant to what could be found on a test at school.

QUALITY STANDARDS:
- Each description should be 100-200 words
- Include concrete examples, not just abstract concepts
- Provide actionable learning content, not just topic overviews

CRITICAL: You MUST return a complete JSON object with ALL required fields. Missing any field will cause the system to fail.

REQUIRED JSON STRUCTURE - Return ONLY this JSON format:
{{
  "units_to_cover": ["Unit 1 Name", "Unit 2 Name", "Unit 3 Name"],
  "pomodoro": "25/5",
  "techniques": ["technique1", "technique2", "technique3"],
  "blocks": [
    {{
      "unit": "Unit 1 Name",
      "techniques": ["quiz", "blurting", "flashcards"],
      "description": "Complete detailed description with key concepts, formulas, examples, and common misconceptions. Should be 100-200 words covering all relevant subtopics for this unit.",
      "duration": {block_duration}
    }},
    {{
      "unit": "Unit 2 Name", 
      "techniques": ["flashcards", "feynman"],
      "description": "Complete detailed description with key concepts, formulas, examples, and common misconceptions. Should be 100-200 words covering all relevant subtopics for this unit.",
      "duration": {block_duration}
    }}
  ]
}}

EXAMPLE COMPLETE RESPONSE:
{{
  "units_to_cover": ["Photosynthesis Overview", "Light Reactions", "Calvin Cycle"],
  "pomodoro": "25/5",
  "techniques": ["feynman", "flashcards", "quiz", "blurting"],
  "blocks": [
    {{
      "unit": "Photosynthesis Overview",
      "techniques": ["feynman", "flashcards"],
      "description": "Photosynthesis converts light energy into chemical energy through two interconnected stages. Master equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Key definitions: autotrophs (self-feeding organisms), chloroplasts (organelles containing chlorophyll), thylakoids (membrane structures for light reactions), stroma (fluid space for Calvin cycle). Critical subtopics: chlorophyll a vs b absorption spectra, stomatal regulation, C3 vs C4 vs CAM pathways, photorespiration effects. Essential principles: light-dependent reactions produce ATP/NADPH, light-independent reactions fix CO2 into glucose, oxygen is a byproduct not the goal. Common errors to avoid: thinking plants don't respire (they do both photosynthesis and respiration), confusing reactants/products, assuming all plant cells photosynthesize (only those with chloroplasts). Quantitative facts: ~1-2% light conversion efficiency, 70% of atmospheric oxygen from photosynthesis, 150 billion tons CO2 fixed annually.",
      "duration": 12
    }},
    {{
      "unit": "Light Reactions",
      "techniques": ["flashcards", "quiz"],
      "description": "Light reactions occur in thylakoid membranes converting light energy to chemical energy. Key equation: 2H2O + 2NADP+ + 3ADP + 3Pi + light → O2 + 2NADPH + 3ATP. Critical components: Photosystem II (P680 reaction center), Photosystem I (P700 reaction center), cytochrome b6f complex, ATP synthase. Essential processes: water splitting (oxygen evolution), electron transport chain, proton pumping, chemiosmosis. Important facts: cyclic vs non-cyclic electron flow, Z-scheme energy diagram, plastoquinone and plastocyanin carriers. Common misconceptions: thinking ATP is made directly by light (it's made by chemiosmosis), confusing photosystems I and II order. Quantitative details: 8 photons needed per O2 molecule, proton gradient of 3-4 pH units, ATP:NADPH ratio of 3:2.",
      "duration": 12
    }}
  ]
}}

Remember: You must include exactly {num_blocks} blocks and ALL required fields (units_to_cover, pomodoro, techniques, blocks) or the system will fail. Each block must have a "techniques" array with 1-3 techniques."""
    
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

def generate_gpt_plan(prompt: str, max_retries: int = 2) -> dict:
    """Generate study plan with GPT with enhanced validation and error handling"""
    
    print(f"📏 Prompt length: {len(prompt)} characters")
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🤖 GPT attempt {attempt + 1}/{max_retries + 1}")
            
            # Try new OpenAI client format first, then fall back to old format
            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert curriculum designer. You MUST return ONLY valid JSON with ALL required fields: units_to_cover, pomodoro, techniques, and blocks. Each block must have a 'techniques' array with 1-3 techniques. Missing any field will cause system failure."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000,  # Reduced from 3000
                    top_p=0.9
                )
                raw_response = completion.choices[0].message.content.strip()
            except AttributeError:
                # Fall back to old format
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert curriculum designer. You MUST return ONLY valid JSON with ALL required fields: units_to_cover, pomodoro, techniques, and blocks. Each block must have a 'techniques' array with 1-3 techniques. Missing any field will cause system failure."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000,
                    top_p=0.9
                )
                raw_response = completion.choices[0].message.content.strip()

            print(f"📝 Raw GPT response length: {len(raw_response)} chars")
            print(f"📝 First 300 chars: {raw_response[:300]}...")
            
            # Clean up response
            if raw_response.startswith("```json"):
                raw_response = raw_response[7:-3].strip()
            elif raw_response.startswith("```"):
                raw_response = raw_response[3:-3].strip()
            
            # Parse JSON
            try:
                parsed = json.loads(raw_response)
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
                print(f"📝 Raw response that failed: {raw_response}")
                raise
            
            # Validate ALL required fields
            required_fields = ["blocks", "units_to_cover", "techniques", "pomodoro"]
            missing_fields = [field for field in required_fields if field not in parsed]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                print(f"📋 Available fields: {list(parsed.keys())}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            blocks = parsed.get("blocks", [])
            if not blocks:
                raise ValueError("No blocks generated")
            
            # Validate block structure
            for i, block in enumerate(blocks):
                required_block_fields = ["unit", "techniques", "description", "duration"]
                missing_block_fields = [field for field in required_block_fields if field not in block]
                if missing_block_fields:
                    print(f"❌ Block {i} missing fields: {missing_block_fields}")
                    raise ValueError(f"Block {i} missing fields: {missing_block_fields}")
                
                # Validate techniques array
                techniques = block.get("techniques", [])
                if not isinstance(techniques, list) or len(techniques) == 0 or len(techniques) > 3:
                    print(f"❌ Block {i} invalid techniques: {techniques}")
                    raise ValueError(f"Block {i} must have 1-3 techniques in array format")
            
            print(f"✅ GPT generated valid response with {len(blocks)} blocks")
            print(f"📊 Units: {len(parsed.get('units_to_cover', []))}")
            print(f"🔧 Techniques: {len(parsed.get('techniques', []))}")
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"🔥 JSON parsing failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to parse GPT response as JSON: {str(e)}")
                
        except Exception as e:
            print(f"🔥 GPT generation failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Max retries exceeded")

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
        parsed = generate_gpt_plan(prompt)
        
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
