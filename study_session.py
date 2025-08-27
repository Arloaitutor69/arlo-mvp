import os
import json
import uuid
import asyncio
import hashlib
from typing import List, Optional, Dict, Tuple, Literal, Any
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

# --- JSON Schema for structured outputs (Pydantic v2 only) --- #
class BlockOutput(BaseModel):
    unit: str
    techniques: List[str]
    description: str
    duration: int

    # Disallow extra keys to ensure items schema has "additionalProperties": false
    model_config = {"extra": "forbid"}

class StudyPlanOutput(BaseModel):
    units_to_cover: List[str]
    pomodoro: Literal["25/5", "30/5", "45/15", "50/10"]
    techniques: List[str]
    blocks: List[BlockOutput]

    # Forbid extras at the root as well
    model_config = {"extra": "forbid"}

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

    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"

    if parsed_summary:
        truncated_summary = parsed_summary[:2500]
        content_section += f"SOURCE MATERIAL TO COVER:\n{truncated_summary}\n\n"

    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    prompt = f"""{content_section}

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

Create a study plan with exactly {num_blocks} blocks of {block_duration} minutes each."""
    return prompt

# --- GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert curriculum designer creating comprehensive study plans. Create study plans with exactly the requested number of blocks. Each block must be educational, actionable, and contain substantial learning content with specific examples, definitions, and key facts.

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \".
4. Use \\n for line breaks within content.
5. No trailing commas.

STUDY PLAN STRUCTURE:
- Each block should fully explain 1-2 subtopics with comprehensive coverage
- Cover all aspects of the requested topic comprehensively  
- Use techniques strategically based on content type
- Include specific details, examples, and key concepts
- Build progressively toward complete mastery

CONTENT QUALITY STANDARDS:
- Each description should be 100-200 words
- Include specific facts, formulas, and examples
- Focus on most important details for the level specified
- Ensure non-overlapping, distinct content across blocks"""

# --- Assistant Examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "units_to_cover": ["Cell Structure and Function", "Cellular Respiration", "Photosynthesis", "Cell Communication"],
  "pomodoro": "25/5",
  "techniques": ["flashcards", "feynman", "quiz", "blurting"],
  "blocks": [
    {
      "unit": "Cell Structure and Function",
      "techniques": ["flashcards", "feynman"],
      "description": "Master fundamental cell components and their roles. 1. Cell membrane - selectively permeable phospholipid bilayer controlling molecular transport. 2. Nucleus - contains DNA, controls cellular activities via transcription. 3. Mitochondria - powerhouse producing ATP through cellular respiration. 4. Ribosomes - protein synthesis sites, free-floating or ER-bound. 5. Endoplasmic reticulum - smooth (lipid synthesis) vs rough (protein modification). 6. Golgi apparatus - protein packaging and modification. 7. Lysosomes - digestive organelles breaking down waste. 8. Cytoskeleton - structural support via microfilaments, microtubules. 9. Prokaryotic vs eukaryotic differences. 10. Cell theory principles.",
      "duration": 15
    },
    {
      "unit": "Cellular Respiration",
      "techniques": ["quiz", "blurting"],
      "description": "Understanding ATP production through glucose breakdown. 1. Overall equation: C₆H₁₂O₆ + 6O₂ → 6CO₂ + 6H₂O + 36-38 ATP. 2. Glycolysis - glucose to pyruvate in cytoplasm, net 2 ATP. 3. Krebs cycle - acetyl-CoA oxidation in mitochondrial matrix, produces NADH, FADH₂. 4. Electron transport chain - chemiosmosis creating proton gradient for ATP synthesis. 5. Aerobic vs anaerobic respiration differences. 6. Fermentation pathways (lactic acid, alcoholic). 7. ATP structure and energy release mechanism. 8. Oxygen's role as final electron acceptor. 9. Location specificity of each stage.",
      "duration": 15
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "units_to_cover": ["Progressive Era Reforms", "Great Depression and New Deal", "World War II Impact"],
  "pomodoro": "30/5", 
  "techniques": ["feynman", "flashcards", "quiz"],
  "blocks": [
    {
      "unit": "Progressive Era Reforms",
      "techniques": ["feynman", "flashcards"],
      "description": "Comprehensive coverage of early 20th century reform movements. 1. Muckrakers - investigative journalists exposing corruption (Ida Tarbell, Upton Sinclair). 2. Trust-busting - Theodore Roosevelt's antitrust actions, Sherman Act enforcement. 3. Pure Food and Drug Act 1906 - response to 'The Jungle'. 4. 16th Amendment - federal income tax authorization. 5. 17th Amendment - direct election of senators. 6. 19th Amendment - women's suffrage victory 1920. 7. Prohibition movement and 18th Amendment. 8. Settlement houses - Jane Addams, Hull House social work. 9. Conservation efforts - national parks, forestry services. 10. Labor reforms - child labor laws, workplace safety.",
      "duration": 20
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "units_to_cover": ["Supply and Demand", "Market Equilibrium", "Elasticity", "Consumer Behavior"],
  "pomodoro": "25/5",
  "techniques": ["quiz", "feynman", "flashcards"],
  "blocks": [
    {
      "unit": "Supply and Demand",
      "techniques": ["quiz", "feynman"],
      "description": "Foundation of market economics and price determination. 1. Law of Demand - inverse relationship between price and quantity demanded, downward sloping curve. 2. Law of Supply - positive relationship between price and quantity supplied, upward sloping curve. 3. Demand shifters - income changes, preferences, substitute/complement prices, expectations, population. 4. Supply shifters - input costs, technology, number of sellers, expectations, government policy. 5. Movement along curves vs curve shifts. 6. Normal vs inferior goods income effects. 7. Substitute and complement relationships. 8. Market demand vs individual demand aggregation. 9. Producer surplus and consumer surplus concepts.",
      "duration": 12
    }
  ]
}
"""

async def update_context_async(payload: dict) -> bool:
    """Asynchronously update context API with better error handling"""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client_http:
            response = await client_http.post(f"{CONTEXT_API}/api/context/update", json=payload)
            return response.status_code == 200
    except Exception as e:
        print(f"⚠️ Context update failed: {e}")
        return False

# --- OpenAI call wrapper --- #
def _call_model_and_get_parsed(input_messages: List[Dict[str, Any]], max_tokens: int = 4000):
    """Call Responses API and parse into StudyPlanOutput (strict schema)."""
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=StudyPlanOutput,
        reasoning={"effort": "low"},
        instructions=(
            "Create comprehensive study plans with exactly the requested number of blocks. "
            "Focus on educational value and comprehensive coverage."
        ),
        max_output_tokens=max_tokens,
    )

def generate_gpt_plan(
    prompt: str,
    objective: Optional[str] = None,
    parsed_summary: Optional[str] = None,
    duration: int = 60
) -> dict:
    """Generate study plan with GPT-5-nano structured outputs. Raises on any failure."""
    print(f"📏 Prompt length: {len(prompt)} characters")
    print(f"🤖 Calling GPT-5-nano...")

    # Messages with assistant examples
    input_messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
        {"role": "user", "content": prompt},
    ]

    # First attempt
    response = _call_model_and_get_parsed(input_messages)

    if getattr(response, "output_parsed", None) is None:
        if hasattr(response, "refusal") and response.refusal:
            raise RuntimeError(f"Model refusal: {response.refusal}")
        # Retry once with a strict correction instruction
        retry_msg = {
            "role": "user",
            "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
        }
        response = _call_model_and_get_parsed(input_messages + [retry_msg])
        if getattr(response, "output_parsed", None) is None:
            raise RuntimeError("Model did not return valid parsed output after retry")

    parsed_output = response.output_parsed

    # Convert to dict format
    result = {
        "units_to_cover": parsed_output.units_to_cover,
        "pomodoro": parsed_output.pomodoro,
        "techniques": parsed_output.techniques,
        "blocks": parsed_output.blocks,
    }

    # Validate blocks are present and structured
    blocks = result.get("blocks", [])
    if not blocks:
        raise RuntimeError("No blocks returned by model")

    for i, block in enumerate(blocks):
        if isinstance(block, BaseModel):
            block = block.model_dump()
        if not isinstance(block, dict):
            raise RuntimeError(f"Block {i} is not a JSON object")
        for key in ("unit", "techniques", "description", "duration"):
            if key not in block:
                raise RuntimeError(f"Block {i} missing required key: {key}")
        if not isinstance(block.get("techniques"), list):
            raise RuntimeError(f"Block {i} 'techniques' must be a list")

    print(f"✅ GPT generated valid response with {len(blocks)} blocks")
    print(f"📊 Units: {len(result.get('units_to_cover', []))}")
    print(f"🔧 Techniques: {len(result.get('techniques', []))}")
    return result

# --- Main Endpoint ---
@router.post("/study-session", response_model=StudyPlanResponse)
async def generate_plan(data: StudyPlanRequest, request: Request):
    """Generate comprehensive study plan with strict erroring on failure."""
    print(f"🚀 Starting study plan generation...")
    print(f"📋 Request: duration={data.duration}, objective={bool(data.objective)}, summary={bool(data.parsed_summary)}")

    try:
        user_id = extract_user_id(request)
        print(f"👤 User ID: {user_id}")

        # Build enhanced prompt
        print("📝 Building enhanced prompt...")
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)

        # Generate plan with GPT (raises on failure)
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
        blocks: List[StudyBlock] = []
        context_tasks = []
        total_time = 0

        for idx, item in enumerate(blocks_json):
            if isinstance(item, BaseModel):
                item = item.model_dump()
            unit = item.get("unit", f"Unit {idx + 1}")
            techniques_list = item.get("techniques", ["feynman"])
            primary_technique = techniques_list[0] if techniques_list else "feynman"
            description = item.get("description", "Study the assigned material")
            duration_block = item.get("duration", 12)
            block_id = f"block_{uuid.uuid4().hex[:8]}"

            study_block = StudyBlock(
                id=block_id,
                unit=unit,
                technique=primary_technique,
                techniques=techniques_list,
                phase=primary_technique,
                tool=primary_technique,
                lovable_component="text-block",
                duration=duration_block,
                description=description,
                position=idx,
            )

            blocks.append(study_block)
            total_time += duration_block

            print(f"✅ Block {idx + 1}: {unit} - {techniques_list} ({duration_block}min)")
            print(f"   Description: {description[:100]}...")

            # Prepare context update (async)
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
                    "review_scheduled": False,
                },
            }
            context_tasks.append(update_context_async(context_payload))

        # Execute context updates with timeout protection (best-effort; failures don't break the endpoint)
        print(f"📤 Sending {len(context_tasks)} context updates...")
        try:
            context_results = await asyncio.wait_for(
                asyncio.gather(*context_tasks, return_exceptions=True),
                timeout=10.0,
            )
            successful_updates = sum(1 for r in context_results if r is True)
            print(f"✅ {successful_updates}/{len(context_tasks)} context updates successful")
        except asyncio.TimeoutError:
            print("⏰ Context updates timed out")

        print(f"🎉 Study plan generated successfully!")
        print(f"📊 Total: {len(blocks)} blocks, {total_time} minutes")

        return StudyPlanResponse(
            session_id=session_id,
            topic=data.objective or "Study Plan from Uploaded Content",
            total_duration=total_time,
            pomodoro=pomodoro,
            units_to_cover=units,
            techniques=techniques,
            blocks=blocks,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Study plan generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
