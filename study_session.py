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
    # Disallow extra keys so items have "additionalProperties": false
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

def _pick_even_divisor(duration: int, candidates: List[int]) -> Tuple[int, int]:
    """
    Choose n in candidates such that duration % n == 0 and block = duration//n
    stays close to 8 minutes (preferred for focused single topics) and within [5, 12].
    Returns (n, block_minutes). Raises if none fits.
    """
    viable = []
    for n in candidates:
        if duration % n == 0:
            block = duration // n
            if 5 <= block <= 12:
                viable.append((abs(block - 8), -n, n, block))  # prefer closer to 8, then larger n
    if not viable:
        raise ValueError(f"No even division found for duration={duration} with allowed block range 5–12.")
    _, _, n, block = sorted(viable)[0]
    return n, block

def calculate_optimal_blocks(duration: int) -> Tuple[int, int]:
    """Calculate optimal number of blocks (7-9 preferred) with focused single-topic durations."""
    
    # First try to get exactly 7, 8, or 9 blocks (preferred range)
    try:
        return _pick_even_divisor(duration, [7, 8, 9])
    except ValueError:
        pass
    
    # If that doesn't work, try other divisors that give us close to 7-9 blocks
    # For 60 minutes: 60/6=10min blocks (6 blocks), 60/10=6min blocks (10 blocks)
    # We'll accept 6-10 blocks as reasonable alternatives
    try:
        return _pick_even_divisor(duration, [6, 10])
    except ValueError:
        pass
    
    # Final fallback - try any reasonable division that allows focused study
    try:
        return _pick_even_divisor(duration, [5, 12, 4, 15])
    except ValueError:
        # If nothing works, create blocks optimized for single topics
        if duration >= 35:  # minimum 5 minutes per block for 7 blocks
            # Prefer 7-9 blocks even with uneven division
            if duration >= 63:  # 9 * 7 minutes
                num_blocks = 9
            elif duration >= 56:  # 8 * 7 minutes  
                num_blocks = 8
            else:
                num_blocks = 7
            block_duration = duration // num_blocks
            return num_blocks, block_duration
        else:
            raise ValueError(f"Duration {duration} too short to create meaningful focused study blocks")

def create_content_hash(objective: str, parsed_summary: str, duration: int) -> str:
    """Create hash for caching purposes"""
    content = f"{objective or ''}{parsed_summary or ''}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt optimized for single-topic focused blocks."""
    num_blocks, block_duration = calculate_optimal_blocks(duration)

    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"

    if parsed_summary:
        truncated_summary = parsed_summary[:2500]
        content_section += f"SOURCE MATERIAL TO COVER:\n{truncated_summary}\n\n"

    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    # --- ENHANCED PROMPT FOR FOCUSED SINGLE-TOPIC BLOCKS ---
    prompt = f"""{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} focused learning blocks  
- Each block should be {block_duration} minutes long
- CRITICAL: Each block must cover ONE specific, narrow topic only

SINGLE-TOPIC FOCUS REQUIREMENT:
- Each block should focus on ONE specific topic
- DO NOT combine multiple major topics in a single block
Examples of GOOD single-topic blocks:
 * Organelles (not “Organelles and Cell Transport”)
 * Photosynthesis (not “Photosynthesis and Cellular Respiration”)
 * European Renaissance (not “Renaissance and Reformation”)
- Break down broad topics into their component parts
- Each block should allow deep, focused study of its specific topic

AVAILABLE TECHNIQUES (choose 1-2 per block based on what's best for each specific topic):
• flashcards: Spaced repetition for memorization of facts, terms, dates
• feynman: Explain concepts in simple terms, teach-back method
• quiz: Active recall testing, self-assessment
• blurting: Free recall without prompts, brain dumps

REQUIREMENTS:
- Each block needs a clear, specific unit/topic name (avoid generic terms)
- Choose 1-2 BEST techniques for each specific narrow topic
- Focus on what will help the student master THIS specific subtopic most effectively
- Each block must cover distinct, non-overlapping content
- Blocks should build logically toward complete understanding
- Ensure comprehensive coverage by breaking subject into {num_blocks} focused components

CONTENT REQUIREMENTS FOR EACH BLOCK:
1. Each description must focus entirely on the single topic specified in the unit name
2. Include: Key definitions specific to this topic, Important formulas/equations for this concept, Specific facts and details relevant to this subtopic only
3. Focus on the most important aspects of THIS specific topic
4. Descriptions should be detailed enough for focused {block_duration}-minute study sessions
5. The collection should cover the subject comprehensively through focused subtopics

TOPIC BREAKDOWN STRATEGY:
- For sciences: Break by specific processes, structures, or mechanisms
- For history: Break by specific events, periods, or causes/effects  
- For literature: Break by themes, characters, literary devices, or chapters
- For math: Break by specific types of problems or concepts
- For languages: Break by grammar rules, verb tenses, or vocabulary themes

Create a study plan with exactly {num_blocks} blocks of {block_duration} minutes each, where each block is laser-focused on mastering one specific subtopic.

DESCRIPTION FORMAT ENFORCEMENT:
- For EACH block's "description", write an ordered, numbered list of **4-6** subtopics that ALL relate to the single main topic
- Format each item as: "<Specific aspect>: <concise explanation with key facts (~8-15 words)>"
- Keep items tightly focused on the block's single topic - no tangential subjects
- Cover the MOST IMPORTANT aspects of this specific topic comprehensively
- Include relevant equations, definitions, examples, or key facts for THIS topic only
- The ENTIRE description should be the numbered list only, staying within 80-150 words total per block
- Ensure every numbered item directly supports understanding of the main topic only"""

    return prompt

# --- GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert curriculum designer creating focused study plans. Create study plans with exactly the requested number of blocks, where each block focuses on ONE specific, narrow topic only. Never combine multiple major concepts into a single block.

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \".
4. Use \\n for line breaks within content.
5. No trailing commas.
6. Each block must have a specific, focused topic name (not generic terms like "Overview" or "Introduction").
7. Break down broad subjects into specific, narrow subtopics for deeper learning.
"""

# --- Assistant Examples (Updated for Single-Topic Focus) --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "units_to_cover": ["Cell Membrane Structure", "Nucleus Organization", "Mitochondria Function", "Ribosome Types", "Endoplasmic Reticulum", "Golgi Apparatus", "Lysosome Activity", "Cytoskeleton Components"],
  "pomodoro": "25/5",
  "techniques": ["flashcards", "feynman", "quiz", "blurting"],
  "blocks": [
    {
      "unit": "Cell Membrane Structure",
      "techniques": ["flashcards", "feynman"],
      "description": "1) Phospholipid bilayer: Hydrophilic heads face outward, hydrophobic tails inward. 2) Membrane proteins: Integral span membrane, peripheral attach surface. 3) Cholesterol: Maintains fluidity, prevents crystallization at low temperatures. 4) Glycoproteins: Carbohydrate chains for cell recognition and signaling. 5) Fluid mosaic model: Dynamic structure with moving components.",
      "duration": 8
    },
    {
      "unit": "Nucleus Organization", 
      "techniques": ["quiz"],
      "description": "1) Nuclear envelope: Double membrane with nuclear pores for transport. 2) Nucleolus: Dense region where ribosomal RNA is synthesized. 3) Chromatin: DNA-protein complex, condenses into chromosomes during division. 4) Nuclear matrix: Protein framework supporting nuclear organization. 5) Nuclear pores: Selective transport of molecules between nucleus and cytoplasm.",
      "duration": 8
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "units_to_cover": ["Progressive Era Muckrakers", "Trust-Busting Legislation", "Pure Food and Drug Act", "Constitutional Amendments 16-19", "Labor Reform Movement", "Urban Settlement Houses", "Conservation Movement"],
  "pomodoro": "30/5", 
  "techniques": ["feynman", "flashcards", "quiz"],
  "blocks": [
    {
      "unit": "Progressive Era Muckrakers",
      "techniques": ["feynman"],
      "description": "1) Ida Tarbell: Exposed Standard Oil monopolistic practices in detailed series. 2) Upton Sinclair: The Jungle revealed unsanitary meatpacking industry conditions. 3) Jacob Riis: How the Other Half Lives documented urban poverty. 4) Lincoln Steffens: The Shame of the Cities exposed municipal corruption. 5) McClure's Magazine: Leading publication platform for investigative journalism.",
      "duration": 9
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "units_to_cover": ["Demand Curves", "Supply Curves", "Market Equilibrium Point", "Price Elasticity", "Income Elasticity", "Consumer Surplus", "Producer Surplus"],
  "pomodoro": "25/5",
  "techniques": ["quiz", "feynman", "flashcards"],
  "blocks": [
    {
      "unit": "Demand Curves",
      "techniques": ["quiz", "feynman"],
      "description": "1) Law of demand: Inverse relationship between price and quantity demanded. 2) Demand shifters: Income, tastes, expectations, related goods prices. 3) Movement vs shift: Along curve vs new curve position. 4) Normal goods: Demand increases with income (positive income elasticity). 5) Inferior goods: Demand decreases as income rises (negative elasticity). 6) Individual vs market: Horizontal summation of all consumers.",
      "duration": 9
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
            "Create focused study plans with exactly the requested number of blocks. "
            "Each block must focus on ONE specific, narrow topic only. "
            "Break down broad subjects into focused subtopics for deeper learning."
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

    input_messages = [
        {"role": "system", "content": GPT_SYSTEM_PROMPT},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
        {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
        {"role": "user", "content": prompt},
    ]

    response = _call_model_and_get_parsed(input_messages)

    if getattr(response, "output_parsed", None) is None:
        if hasattr(response, "refusal") and response.refusal:
            raise RuntimeError(f"Model refusal: {response.refusal}")
        retry_msg = {
            "role": "user",
            "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Focus on single-topic blocks only."
        }
        response = _call_model_and_get_parsed(input_messages + [retry_msg])
        if getattr(response, "output_parsed", None) is None:
            raise RuntimeError("Model did not return valid parsed output after retry")

    parsed_output = response.output_parsed

    result = {
        "units_to_cover": parsed_output.units_to_cover,
        "pomodoro": parsed_output.pomodoro,
        "techniques": parsed_output.techniques,
        "blocks": parsed_output.blocks,
    }

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

    print(f"✅ GPT generated valid response with {len(blocks)} focused blocks")
    print(f"📊 Units: {len(result.get('units_to_cover', []))}")
    print(f"🔧 Techniques: {len(result.get('techniques', []))}")
    return result

# --- Main Endpoint ---
@router.post("/study-session", response_model=StudyPlanResponse)
async def generate_plan(data: StudyPlanRequest, request: Request):
    """Generate comprehensive study plan with focused single-topic blocks."""
    print(f"🚀 Starting focused study plan generation...")
    print(f"📋 Request: duration={data.duration}, objective={bool(data.objective)}, summary={bool(data.parsed_summary)}")

    try:
        user_id = extract_user_id(request)
        print(f"👤 User ID: {user_id}")

        print("📝 Building enhanced single-topic focused prompt...")
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)

        print("🤖 Calling GPT...")
        parsed = generate_gpt_plan(prompt, data.objective, data.parsed_summary, data.duration)

        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")

        print(f"📦 Processing {len(blocks_json)} focused blocks...")

        blocks: List[StudyBlock] = []
        context_tasks = []
        total_time = 0

        for idx, item in enumerate(blocks_json):
            if isinstance(item, BaseModel):
                item = item.model_dump()
            unit = item.get("unit", f"Topic {idx + 1}")
            techniques_list = item.get("techniques", ["feynman"])
            primary_technique = techniques_list[0] if techniques_list else "feynman"
            description = item.get("description", "1) Key concept: Main point. 2) Important detail: Supporting information.")
            duration_block = item.get("duration", 8)
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

            context_payload = {
                "source": "session_planner",
                "user_id": user_id,
                "current_topic": f"{unit} — {primary_technique}",
                "learning_event": {
                    "concept": unit,
                    "phase": primary_technique,
                    "confidence": None,
                    "depth": None,
                    "source_summary": f"Planned focused {' + '.join(techniques_list)} session: {description[:200]}...",
                    "repetition_count": 0,
                    "review_scheduled": False,
                },
            }
            context_tasks.append(update_context_async(context_payload))

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

        print(f"🎉 Focused study plan generated successfully!")
        print(f"📊 Total: {len(blocks)} focused blocks, {total_time} minutes")

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
