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
    stays close to 10 minutes (preferred) and within [6, 15].
    Returns (n, block_minutes). Raises if none fits.
    """
    viable = []
    for n in candidates:
        if duration % n == 0:
            block = duration // n
            if 6 <= block <= 15:
                viable.append((abs(block - 10), -n, n, block))  # prefer closer to 10, then larger n
    if not viable:
        raise ValueError(f"No even division found for duration={duration} with allowed block range 6–15.")
    _, _, n, block = sorted(viable)[0]
    return n, block

def calculate_optimal_blocks(duration: int) -> Tuple[int, int]:
    """Calculate optimal number of blocks (7-8 preferred) with flexible duration ranges."""
    
    # First try to get exactly 7 or 8 blocks
    try:
        return _pick_even_divisor(duration, [7, 8])
    except ValueError:
        pass
    
    # If that doesn't work, try other divisors that give us close to 7-8 blocks
    # For 60 minutes: 60/6=10min blocks (6 blocks), 60/10=6min blocks (10 blocks)
    # We'll accept 6-10 blocks as reasonable alternatives
    try:
        return _pick_even_divisor(duration, [6, 9, 10])
    except ValueError:
        pass
    
    # Final fallback - try any reasonable division
    try:
        return _pick_even_divisor(duration, [5, 4, 12, 15])
    except ValueError:
        # If nothing works, just create 7 blocks with uneven division
        if duration >= 42:  # minimum 6 minutes per block
            block_duration = duration // 7
            return 7, block_duration
        else:
            raise ValueError(f"Duration {duration} too short to create meaningful study blocks")

def create_content_hash(objective: str, parsed_summary: str, duration: int) -> str:
    """Create hash for caching purposes"""
    content = f"{objective or ''}{parsed_summary or ''}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt with better structure and examples (appended format spec)."""
    num_blocks, block_duration = calculate_optimal_blocks(duration)

    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"

    if parsed_summary:
        truncated_summary = parsed_summary[:2500]
        content_section += f"SOURCE MATERIAL TO COVER:\n{truncated_summary}\n\n"

    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    # --- ORIGINAL PROMPT (kept intact) ---
    prompt = f"""{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} learning blocks  
- Each block should be {block_duration} minutes long

AVAILABLE TECHNIQUES (choose 1-2 per block based on what's best for each unit):
• flashcards: Spaced repetition for memorization
• feynman: Explain concepts in simple terms
• quiz: Active recall testing
• blurting: Free recall without prompts

REQUIREMENTS:
- Each block needs a clear unit/topic name
- Choose 1-2 BEST techniques for each specific unit/topic - focus on quality over quantity 
- You can use any combination or sequence of techniques within a block
- Focus on what will help the student learn THIS specific content most effectively
- Each block must cover distinct, non redundant and non-overlapping content that builds progressively toward complete mastery of the subject

CONTENT REQUIREMENTS FOR EACH BLOCK:
1. Each description must be a complete self contained lesson including:
2. Key definitions and examples, Important formulas, equations, or principles, Specific facts, data points, or details to remember
3. focus on most important details as oppose to minor tangential. Cater to highschool/undergraduate college info unless otherwise specified
4. the collection of study topics and descriptions should cover the entirety of the content the student wants to learn in that session

Create a study plan with exactly {num_blocks} blocks of {block_duration} minutes each."""

    # --- NEW APPENDED SECTION (does not remove/alter anything above) ---
    # Drives the model to produce the numbered-list descriptions like your examples.
    prompt += f"""

DESCRIPTION FORMAT ENFORCEMENT (MATCH EXAMPLES):
- For EACH block's "description", write an ordered, numbered list of **5-6** subtopics.
- Format each item as: "<Short subtopic title>: <very concise teacher note (~5–12 words), may include 1 parenthetical fact/date/case>".
- Keep the list tight and factual; avoid paragraphs or narrative prose.
- Cover the MOST IMPORTANT concepts for the unit end-to-end; items should be **non-overlapping** and **collectively exhaustive** for the subtopic.
- Include, when relevant, key legislation, court cases, dates, turning points, equations or definitions.
- The ENTIRE description should be the numbered list only (no intro/outro text), staying within 100–200 words total per block.
"""

    return prompt

# --- GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert curriculum designer creating comprehensive study plans. Create study plans with exactly the requested number of blocks. Each block must be educational, actionable, and contain substantial learning content with specific examples, definitions, and key facts.

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \".
4. Use \\n for line breaks within content.
5. No trailing commas.
"""

# --- Assistant Examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "units_to_cover": ["Cell Structure and Function", "Cellular Respiration", "Photosynthesis", "Cell Communication", "Cell Division", "Genetics Basics", "Protein Synthesis"],
  "pomodoro": "25/5",
  "techniques": ["flashcards", "feynman", "quiz", "blurting"],
  "blocks": [
    {
      "unit": "Cell Structure and Function",
      "techniques": ["flashcards", "feynman"],
      "description": "1) Cell membrane: Selective bilayer; transport; signaling. 2) Nucleus: DNA control; transcription; nucleolus. 3) Mitochondria: ATP production; aerobic metabolism. 4) Ribosomes: Protein synthesis; free vs bound. 5) Smooth ER: Lipids; detox. 6) Rough ER: Protein modification. 7) Golgi: Packaging; vesicles. 8) Lysosomes: Digestive enzymes; waste removal. 9) Cytoskeleton: Microtubules/microfilaments; structure. 10) Prokaryotes vs eukaryotes: Organization; size; nuclei. 11) Cell theory: All cells from cells; fundamental unit. 12) Membrane proteins: Channels; receptors.",
      "duration": 9
    },
    {
      "unit": "Cellular Respiration",
      "techniques": ["quiz"],
      "description": "1) Equation: C6H12O6+O2→CO2+H2O+ATP. 2) Glycolysis: Cytoplasm; 2 ATP net. 3) Link reaction: Pyruvate→Acetyl-CoA. 4) Krebs: Matrix; NADH/FADH2 yield. 5) ETC: Proton gradient; ATP synthase. 6) Oxygen: Final electron acceptor. 7) Anaerobic: Lactic/alcoholic fermentation. 8) ATP structure: Phosphate bonds; energy. 9) Regulation: Allosteric enzymes. 10) Location: Stages by organelle. 11) Energy balance: 30–32 ATP. 12) Intermediates: Citrate, oxaloacetate.",
      "duration": 9
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "units_to_cover": ["Progressive Era Reforms", "Great Depression and New Deal", "World War II Impact", "Cold War Origins", "Civil Rights Movement", "Modern America", "Contemporary Issues"],
  "pomodoro": "30/5", 
  "techniques": ["feynman", "flashcards", "quiz"],
  "blocks": [
    {
      "unit": "Progressive Era Reforms",
      "techniques": ["feynman"],
      "description": "1) Muckrakers: Tarbell, Sinclair expose abuses. 2) Trust-busting: Sherman/Clayton enforcement. 3) Food/Drug Safety: 1906 acts modernize standards. 4) 16th–19th Amendments: Tax, senators, suffrage. 5) Labor: Child labor laws; safety rules. 6) Cities: Settlement houses; Hull House. 7) Conservation: Parks; forestry service. 8) Direct democracy: Initiative, referendum, recall. 9) Corporate regulation: ICC/Federal Trade. 10) State reforms: Wisconsin Idea. 11) Education: Compulsory schooling expands. 12) Courts: Lochner era limits.",
      "duration": 8
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "units_to_cover": ["Supply and Demand", "Market Equilibrium", "Elasticity", "Consumer Behavior", "Producer Theory", "Market Structures", "Government Intervention"],
  "pomodoro": "25/5",
  "techniques": ["quiz", "feynman", "flashcards"],
  "blocks": [
    {
      "unit": "Supply and Demand",
      "techniques": ["quiz", "feynman"],
      "description": "1) Law of demand: Inverse P-Qd. 2) Law of supply: Positive P-Qs. 3) Shifters: Income, tastes, expectations, related goods. 4) Supply shifters: Inputs, tech, sellers, policy. 5) Movement vs shift: Along vs new curve. 6) Normal vs inferior: Income responses. 7) Substitutes/complements: Cross-price effects. 8) Market demand: Horizontal sum of individuals. 9) Surplus/shortage: Price signals. 10) Consumer surplus: Value minus price. 11) Producer surplus: Price minus cost. 12) Welfare: Deadweight loss.",
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
            "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
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

        print("📝 Building enhanced prompt...")
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)

        print("🤖 Calling GPT...")
        parsed = generate_gpt_plan(prompt, data.objective, data.parsed_summary, data.duration)

        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")

        print(f"📦 Processing {len(blocks_json)} blocks...")

        blocks: List[StudyBlock] = []
        context_tasks = []
        total_time = 0

        for idx, item in enumerate(blocks_json):
            if isinstance(item, BaseModel):
                item = item.model_dump()
            unit = item.get("unit", f"Unit {idx + 1}")
            techniques_list = item.get("techniques", ["feynman"])
            primary_technique = techniques_list[0] if techniques_list else "feynman"
            description = item.get("description", "1) Topic: Key point. 2) Topic: Key point.")  # safe minimal list
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
                    "source_summary": f"Planned {' + '.join(techniques_list)} session: {description[:200]}...",
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
