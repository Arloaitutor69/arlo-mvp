import os
import json
import uuid
import asyncio
import hashlib
import re
from typing import List, Optional, Dict, Tuple
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
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

# --- JSON Parsing Utilities ---
def fix_json_escaping(text):
    """Fix common JSON escaping issues - borrowed from teaching.py"""
    # Remove any control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Fix single quotes that should be escaped double quotes
    text = re.sub(r"(?<!\\)'", '"', text)
    
    # Fix unescaped quotes inside content strings
    def fix_content_quotes(match):
        content_part = match.group(1)
        # Escape any unescaped quotes inside the content
        content_part = re.sub(r'(?<!\\)"', r'\\"', content_part)
        return f'"description": "{content_part}"'
    
    # Apply the fix to description fields
    text = re.sub(r'"description":\s*"([^"]*(?:\\.[^"]*)*)"', fix_content_quotes, text)
    
    # Remove trailing commas before closing brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text

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
Each description must be a complete self contained lesson including:
1. Key definitions and examples, Important formulas, equations, or principles, Specific facts, data points, or details to remember
2. focus on most important details as oppose to minor tangential. Cater to highschool/undergraduate college info unless otherwise specified

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
      "description": "content here",
      "duration": {block_duration}
    }},
    {{
      "unit": "Unit 2 Name", 
      "techniques": ["flashcards", "feynman"],
      "description": "content here",
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
      "description": "Photosynthesis converts light energy into chemical energy through two interconnected stages. Master equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Key definitions: autotrophs (self-feeding organisms), chloroplasts (organelles containing chlorophyll), thylakoids (membrane structures for light reactions), stroma (fluid space for Calvin cycle). Critical subtopics: chlorophyll a vs b absorption spectra, stomatal regulation, C3 vs C4 vs CAM pathways, photorespiration effects. Essential principles: light-dependent reactions produce ATP/NADPH, light-independent reactions fix CO2 into glucose, oxygen is a byproduct not the goal. Quantitative facts: ~1-2% light conversion efficiency, 70% of atmospheric oxygen from photosynthesis, 150 billion tons CO2 fixed annually.",
      "duration": {block_duration}
    }},
    {{
      "unit": "Light Reactions",
      "techniques": ["flashcards", "quiz"],
      "description": "Light reactions occur in thylakoid membranes converting light energy to chemical energy. Key equation: 2H2O + 2NADP+ + 3ADP + 3Pi + light → O2 + 2NADPH + 3ATP. Critical components: Photosystem II (P680 reaction center), Photosystem I (P700 reaction center), cytochrome b6f complex, ATP synthase. Essential processes: water splitting (oxygen evolution), electron transport chain, proton pumping, chemiosmosis. Important facts: cyclic vs non-cyclic electron flow, Z-scheme energy diagram, plastoquinone and plastocyanin carriers. Quantitative details: 8 photons needed per O2 molecule, proton gradient of 3-4 pH units, ATP:NADPH ratio of 3:2.",
      "duration": {block_duration}
    }}
  ]
}}

Remember: You must include exactly {num_blocks} blocks and ALL required fields (units_to_cover, pomodoro, techniques, blocks) or the system will fail. Each block must have a "techniques" array with 1-3 techniques. Output ONLY valid JSON format with proper escaping."""
    
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
    """Generate study plan with GPT-5-nano with enhanced validation and error handling"""
    
    print(f"📏 Prompt length: {len(prompt)} characters")
    
    try:
        print(f"🤖 Calling GPT-5-nano...")
        
        # Use new GPT-5-nano API call format from teaching.py
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are an expert curriculum designer. You MUST return ONLY valid JSON with ALL required fields: units_to_cover, pomodoro, techniques, and blocks. Each block must have a 'techniques' array with 1-3 techniques. Missing any field will cause system failure."},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="low"
        )

        raw_output = response.choices[0].message.content.strip()
        
        print(f"📝 Raw GPT response length: {len(raw_output)} chars")
        print(f"📝 First 300 chars: {raw_output[:300]}...")
        
        # Remove code block markers
        if raw_output.startswith("```"):
            raw_output = re.sub(r'^```(?:json)?\n?', '', raw_output)
            raw_output = re.sub(r'\n?```$', '', raw_output)
        
        # Fix JSON escaping issues
        raw_output = fix_json_escaping(raw_output)
        
        # Parse JSON with better error handling
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw output sample: {raw_output[:200]}...")
            # Use fallback response
            parsed_output = create_fallback_study_plan(objective, parsed_summary, duration)
        
        # Validate ALL required fields
        required_fields = ["blocks", "units_to_cover", "techniques", "pomodoro"]
        missing_fields = [field for field in required_fields if field not in parsed_output]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            print(f"📋 Available fields: {list(parsed_output.keys())}")
            # Use fallback response
            parsed_output = create_fallback_study_plan(objective, parsed_summary, duration)
        
        blocks = parsed_output.get("blocks", [])
        if not blocks:
            print("❌ No blocks generated, using fallback")
            parsed_output = create_fallback_study_plan(objective, parsed_summary, duration)
            blocks = parsed_output.get("blocks", [])
        
        # Validate and clean block structure
        num_blocks, block_duration = calculate_optimal_blocks(duration)
        valid_techniques = {"flashcards", "feynman", "quiz", "blurting"}
        
        # Ensure exactly the right number of blocks
        if len(blocks) != num_blocks:
            if len(blocks) < num_blocks:
                # Pad with additional blocks if needed
                while len(blocks) < num_blocks:
                    blocks.append({
                        "unit": f"Additional Study {len(blocks) + 1}",
                        "techniques": ["feynman", "flashcards"],
                        "description": f"Additional study content to complete the {duration}-minute session.",
                        "duration": block_duration
                    })
            else:
                # Trim to exactly the right number
                blocks = blocks[:num_blocks]
        
        # Clean and validate each block
        for i, block in enumerate(blocks):
            # Ensure block is a dictionary
            if not isinstance(block, dict):
                block = {
                    "unit": f"Study Block {i+1}",
                    "techniques": ["feynman"],
                    "description": "Study the assigned material using active learning techniques.",
                    "duration": block_duration
                }
                blocks[i] = block

            # Validate required block fields
            required_block_fields = ["unit", "techniques", "description", "duration"]
            for field in required_block_fields:
                if field not in block:
                    if field == "unit":
                        block[field] = f"Study Block {i+1}"
                    elif field == "techniques":
                        block[field] = ["feynman"]
                    elif field == "description":
                        block[field] = "Study the assigned material using active learning techniques."
                    elif field == "duration":
                        block[field] = block_duration

            # Validate techniques array
            techniques = block.get("techniques", [])
            if not isinstance(techniques, list) or len(techniques) == 0 or len(techniques) > 3:
                block["techniques"] = ["feynman"]
            else:
                # Ensure all techniques are valid
                block["techniques"] = [t for t in techniques if t in valid_techniques]
                if not block["techniques"]:
                    block["techniques"] = ["feynman"]

            # Ensure description is string and clean it
            description = block.get("description", "")
            if isinstance(description, list):
                description = "\\n\\n".join([str(item) for item in description])
            elif not isinstance(description, str):
                description = str(description)
            
            # Clean description string
            description = description.replace('\n', '\\n').replace('\r', '')
            block["description"] = description

            # Ensure duration is integer
            if not isinstance(block.get("duration"), int):
                block["duration"] = block_duration
        
        # Update the cleaned blocks back to parsed_output
        parsed_output["blocks"] = blocks
        
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
