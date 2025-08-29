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

def calculate_optimal_blocks(duration: int) -> Tuple[int, int]:
    """
    Calculate optimal number of blocks based on duration targets:
    - 60 minutes → 7-8 blocks (7.5-8.5 min each)
    - 90 minutes → 9-10 blocks (9-10 min each) 
    - 120 minutes → 12 blocks (10 min each)
    
    Returns (num_blocks, block_duration_minutes)
    """
    
    # Define target ranges for different durations
    if duration <= 45:
        # Short sessions: 6-7 blocks, 6-7.5 min each
        target_blocks = [6, 7]
        min_block_duration, max_block_duration = 6, 8
    elif duration <= 65:
        # ~60 min sessions: 7-8 blocks, 7.5-8.5 min each
        target_blocks = [7, 8]
        min_block_duration, max_block_duration = 7, 9
    elif duration <= 95:
        # ~90 min sessions: 9-10 blocks, 9-10 min each
        target_blocks = [9, 10]
        min_block_duration, max_block_duration = 8, 11
    elif duration <= 125:
        # ~120 min sessions: 12 blocks, ~10 min each
        target_blocks = [12, 11]
        min_block_duration, max_block_duration = 9, 12
    else:
        # Very long sessions: scale appropriately
        target_blocks = [max(12, duration // 10), max(14, duration // 9)]
        min_block_duration, max_block_duration = 9, 12
    
    # Try target blocks first
    for num_blocks in target_blocks:
        if duration % num_blocks == 0:
            block_duration = duration // num_blocks
            if min_block_duration <= block_duration <= max_block_duration:
                print(f"📏 Perfect division: {num_blocks} blocks of {block_duration} minutes each")
                return num_blocks, block_duration
    
    # If perfect division doesn't work, find best approximate division
    best_option = None
    best_score = float('inf')
    
    # Try a wider range around target
    all_candidates = list(range(max(4, min(target_blocks) - 2), max(target_blocks) + 3))
    
    for num_blocks in all_candidates:
        block_duration = duration // num_blocks
        remainder = duration % num_blocks
        
        # Score based on how close we are to ideal block duration and minimal remainder
        if min_block_duration <= block_duration <= max_block_duration:
            # Prefer solutions with no remainder, then minimize remainder
            score = remainder * 10 + abs(num_blocks - target_blocks[0])
            if score < best_score:
                best_score = score
                best_option = (num_blocks, block_duration)
    
    if best_option:
        num_blocks, block_duration = best_option
        print(f"📏 Optimal division: {num_blocks} blocks of {block_duration} minutes each")
        return num_blocks, block_duration
    
    # Fallback: use closest target
    num_blocks = target_blocks[0]
    block_duration = duration // num_blocks
    print(f"📏 Fallback division: {num_blocks} blocks of {block_duration} minutes each")
    return num_blocks, block_duration

def create_content_hash(objective: str, parsed_summary: str, duration: int) -> str:
    """Create hash for caching purposes"""
    content = f"{objective or ''}{parsed_summary or ''}{duration}"
    return hashlib.md5(content.encode()).hexdigest()

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt optimized for appropriately-granular focused blocks."""
    num_blocks, block_duration = calculate_optimal_blocks(duration)

    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"

    if parsed_summary:
        truncated_summary = parsed_summary[:2500]
        content_section += f"SOURCE MATERIAL TO COVER:\n{truncated_summary}\n\n"

    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    # --- ENHANCED PROMPT FOR APPROPRIATELY-GRANULAR FOCUSED BLOCKS ---
    prompt = f"""{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} focused learning blocks  
- Each block should be {block_duration} minutes long
- CRITICAL: Each block must cover ONE substantial, complete topic

TOPIC GRANULARITY REQUIREMENTS:
Each block should cover a COMPLETE, SUBSTANTIAL topic that is:
✅ GOOD EXAMPLES (substantial but focused):
   • "Causes of the Renaissance" (not just "Economic factors")
   • "Key Renaissance Figures" (not just "Leonardo da Vinci")
   • "Photosynthesis Process" (not just "Light reactions")
   • "World War I Battles" (not just "Battle of Verdun")
   • "Cell Division Process" (not just "Prophase")
   • "Supply and Demand" (not just "Demand curves")

❌ AVOID (too narrow/specific):
   • Individual people (unless they're the entire subject)
   • Single events within larger processes
   • Individual steps of multi-step processes
   • Single examples of broader concepts

TOPIC SCOPE GUIDELINES:
- Each topic should take a full {block_duration} minutes of focused study
- Topics should be substantial enough to have multiple key aspects to explore
- Break down broad subjects into their major component themes/areas
- Ensure each block represents a complete learning unit that builds understanding
- Topics should be cohesive - all aspects should relate to the same central concept

AVAILABLE TECHNIQUES (choose 1-2 per block based on what's best for each specific topic):
• flashcards: Spaced repetition for memorization of facts, terms, dates
• feynman: Explain concepts in simple terms, teach-back method  
• quiz: Active recall testing, self-assessment
• blurting: Free recall without prompts, brain dumps

REQUIREMENTS:
- Each block needs a clear, substantial topic name that indicates complete learning unit
- Choose 1-2 BEST techniques for each topic based on its nature
- Focus on what will help student master this complete topic most effectively
- Each block must cover distinct, substantial content areas
- Blocks should build logically toward complete subject mastery
- Ensure comprehensive coverage by breaking subject into {num_blocks} substantial components

CONTENT REQUIREMENTS FOR EACH BLOCK:
1. Each description must focus entirely on the substantial topic specified in unit name
2. Include: Key definitions, concepts, and principles for this topic area
3. Important facts, examples, and applications relevant to this complete topic
4. Specific details that support comprehensive understanding of this topic
5. Focus on the most important aspects needed to master this complete topic area
6. Descriptions should support focused {block_duration}-minute study sessions
7. The collection should cover the subject comprehensively through substantial topic areas

DESCRIPTION FORMAT ENFORCEMENT:
- For EACH block's "description", write an ordered, numbered list of **4-6** key aspects
- Format: "<Major aspect>: <comprehensive explanation with key details (~12-20 words)>"
- Keep items focused on the block's substantial topic - comprehensive coverage of that topic
- Cover the MOST IMPORTANT aspects needed to master this complete topic area
- Include relevant equations, definitions, examples, key facts for THIS topic comprehensively
- The ENTIRE description should be numbered list only, 100-180 words total per block
- Ensure every item supports complete understanding of the substantial topic

Create a study plan with exactly {num_blocks} blocks of {block_duration} minutes each, where each block focuses on mastering one substantial, complete topic area that requires the full time allocation."""

    return prompt

# --- GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert curriculum designer creating focused study plans. Create study plans with exactly the requested number of blocks, where each block focuses on ONE substantial, complete topic that warrants the full time allocation.

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \".
4. Use \\n for line breaks within content.
5. No trailing commas.
6. Each block must cover a substantial, complete topic (not overly narrow sub-components).
7. Topics should be significant enough to justify the allocated study time.
8. Break subjects into major component areas, themes, or processes - not individual details.
"""

# --- Assistant Examples (Updated for Appropriate Topic Granularity) --- #
# --- Assistant Examples (Balanced Units, Subtopics + Notes Only) --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "units_to_cover": [
    "Cell Structure and Function",
    "Cell Transport and Homeostasis",
    "Cellular Energy Production",
    "Cell Division and Genetic Continuity",
    "Gene Expression and Protein Synthesis",
    "Cell Communication and Signaling",
    "Cellular Waste and Defense Mechanisms"
  ],
  "pomodoro": "25/5",
  "techniques": ["flashcards", "feynman", "quiz", "blurting"],
  "blocks": [
    {
      "unit": "Cell Structure and Function",
      "techniques": ["flashcards", "feynman"],
      "description": "1) Cell membrane – structure, permeability. 2) Nucleus – DNA storage, nucleolus. 3) Organelles – mitochondria, ER, Golgi. 4) Cytoskeleton – filaments, structural roles. 5) Compartmentalization – specialized functions.",
      "duration": 8
    },
    {
      "unit": "Cell Transport and Homeostasis",
      "techniques": ["quiz", "feynman"],
      "description": "1) Passive transport – diffusion, osmosis. 2) Active transport – ATP, pumps. 3) Bulk transport – endocytosis, exocytosis. 4) Channels – ion channels, aquaporins. 5) Regulation – maintaining equilibrium.",
      "duration": 8
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "units_to_cover": [
    "Origins of the Renaissance",
    "Renaissance Art and Humanism",
    "Political and Economic Shifts in Renaissance Europe",
    "Key Figures of the Renaissance",
    "Scientific and Intellectual Developments",
    "Origins of the Protestant Reformation",
    "Spread of the Reformation and Religious Conflict",
    "Legacy of the Renaissance and Reformation"
  ],
  "pomodoro": "30/5",
  "techniques": ["feynman", "flashcards", "quiz"],
  "blocks": [
    {
      "unit": "Origins of the Renaissance",
      "techniques": ["feynman", "quiz"],
      "description": "1) Economic foundations – trade, banking. 2) Social factors – urbanization, feudal decline. 3) Classical revival – humanism. 4) Political conditions – Italian city-states. 5) Cultural exchange – Byzantine/Islamic influences.",
      "duration": 9
    },
    {
      "unit": "Origins of the Protestant Reformation",
      "techniques": ["feynman", "flashcards"],
      "description": "1) Criticisms of Catholic Church – indulgences, corruption. 2) Martin Luther – 95 Theses, justification by faith. 3) Printing press – spread of ideas. 4) Early reformers – Wycliffe, Hus. 5) Church response – Counter-Reformation context.",
      "duration": 9
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "units_to_cover": [
    "Supply and Demand",
    "Market Equilibrium and Changes",
    "Elasticity and Market Sensitivity",
    "Consumer Choice Theory",
    "Producer Decision-Making",
    "Market Efficiency and Failures",
    "Government Intervention in Markets"
  ],
  "pomodoro": "25/5",
  "techniques": ["quiz", "feynman", "flashcards"],
  "blocks": [
    {
      "unit": "Supply and Demand",
      "techniques": ["quiz", "feynman"],
      "description": "1) Law of demand – relationship with price. 2) Demand determinants – income, substitutes, complements. 3) Law of supply – relationship with price. 4) Supply determinants – costs, technology, policy. 5) Shifts vs movements – clear distinction.",
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
            "Each block must focus on ONE substantial, complete topic that justifies the allocated time. "
            "Break subjects into major component areas, not overly narrow sub-details."
        ),
        max_output_tokens=max_tokens
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
            "content": "Fix JSON only: Return corrected JSON with substantial, complete topics (not overly narrow). Each topic should justify the full allocated time."
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

    print(f"✅ GPT generated valid response with {len(blocks)} substantial topic blocks")
    print(f"📊 Units: {len(result.get('units_to_cover', []))}")
    print(f"🔧 Techniques: {len(result.get('techniques', []))}")
    return result

# --- Main Endpoint ---
@router.post("/study-session", response_model=StudyPlanResponse)
async def generate_plan(data: StudyPlanRequest, request: Request):
    """Generate comprehensive study plan with appropriately-granular focused blocks."""
    print(f"🚀 Starting focused study plan generation...")
    print(f"📋 Request: duration={data.duration}, objective={bool(data.objective)}, summary={bool(data.parsed_summary)}")

    try:
        user_id = extract_user_id(request)
        print(f"👤 User ID: {user_id}")

        print("📝 Building enhanced appropriately-granular focused prompt...")
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)

        print("🤖 Calling GPT...")
        parsed = generate_gpt_plan(prompt, data.objective, data.parsed_summary, data.duration)

        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")

        print(f"📦 Processing {len(blocks_json)} substantial topic blocks...")

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
        print(f"📊 Total: {len(blocks)} substantial topic blocks, {total_time} minutes")

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
