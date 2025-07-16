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

class StudyTechnique(BaseModel):
    name: str
    sequence: int  # Order within the block (1, 2, 3)
    duration: int  # Minutes for this technique
    description: str  # What to do with this technique

class StudyBlock(BaseModel):
    id: str
    unit: str
    techniques: List[StudyTechnique]  # Multiple techniques per block
    phase: str  # Primary phase/category
    tool: str  # Primary tool
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

# --- Technique Grouping Logic ---
TECHNIQUE_CATEGORIES = {
    "memorization": ["flashcards", "spaced_repetition", "mnemonics"],
    "conceptual": ["feynman", "analogies", "mind_mapping"],
    "application": ["practice_problems", "case_studies", "simulation"],
    "assessment": ["quiz", "self_test", "peer_review"],
    "recall": ["blurting", "free_recall", "active_recall"]
}

EFFECTIVE_TECHNIQUE_SEQUENCES = {
    "memorization_heavy": [
        ["flashcards", "spaced_repetition", "quiz"],
        ["flashcards", "blurting", "quiz"],
        ["mnemonics", "flashcards", "self_test"]
    ],
    "conceptual_heavy": [
        ["feynman", "analogies", "quiz"],
        ["mind_mapping", "feynman", "self_test"],
        ["feynman", "practice_problems", "peer_review"]
    ],
    "mixed_content": [
        ["flashcards", "feynman", "quiz"],
        ["blurting", "analogies", "self_test"],
        ["mind_mapping", "practice_problems", "quiz"]
    ],
    "application_heavy": [
        ["practice_problems", "case_studies", "quiz"],
        ["simulation", "practice_problems", "peer_review"],
        ["feynman", "practice_problems", "self_test"]
    ]
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

def determine_content_type(unit_description: str) -> str:
    """Analyze content to determine optimal technique category"""
    unit_lower = unit_description.lower()
    
    # Keywords that indicate different types of content
    memorization_keywords = ["formula", "equation", "definition", "term", "vocab", "fact", "date", "name"]
    conceptual_keywords = ["concept", "theory", "principle", "understand", "explain", "process", "system"]
    application_keywords = ["problem", "calculate", "solve", "apply", "example", "practice", "exercise"]
    
    memorization_score = sum(1 for keyword in memorization_keywords if keyword in unit_lower)
    conceptual_score = sum(1 for keyword in conceptual_keywords if keyword in unit_lower)
    application_score = sum(1 for keyword in application_keywords if keyword in unit_lower)
    
    if memorization_score > conceptual_score and memorization_score > application_score:
        return "memorization_heavy"
    elif application_score > conceptual_score and application_score > memorization_score:
        return "application_heavy"
    elif conceptual_score > 0:
        return "conceptual_heavy"
    else:
        return "mixed_content"

def build_enhanced_prompt(objective: Optional[str], parsed_summary: Optional[str], duration: int) -> str:
    """Build comprehensive GPT prompt with multi-technique support"""
    
    num_blocks, block_duration = calculate_optimal_blocks(duration)
    
    # Build content section
    content_section = ""
    if objective:
        content_section += f"STUDENT'S LEARNING OBJECTIVE:\n{objective.strip()}\n\n"
    
    if parsed_summary:
        content_section += f"SOURCE MATERIAL TO COVER:\n{parsed_summary[:4500]}\n\n"
    
    if not objective and not parsed_summary:
        raise ValueError("At least one of objective or parsed_summary must be provided.")

    prompt = f"""You are an expert curriculum designer creating a study plan with intelligent technique sequencing.

{content_section}

PLAN SPECIFICATIONS:
- Duration: {duration} minutes total
- Create exactly {num_blocks} learning blocks  
- Each block should be {block_duration} minutes long
- Each block should use 2-3 complementary techniques in sequence

AVAILABLE TECHNIQUES WITH OPTIMAL USAGE:

MEMORIZATION TECHNIQUES:
• flashcards: Spaced repetition for facts, formulas, definitions (3-5 min)
• spaced_repetition: Review previously learned material (2-4 min)
• mnemonics: Memory aids for complex lists or sequences (2-3 min)

CONCEPTUAL TECHNIQUES:
• feynman: Explain concepts in simple terms (4-6 min)
• analogies: Connect new concepts to familiar ones (2-4 min)
• mind_mapping: Visual concept connections (3-5 min)

APPLICATION TECHNIQUES:
• practice_problems: Work through examples and exercises (5-8 min)
• case_studies: Real-world applications (4-6 min)
• simulation: Mental or physical modeling (3-5 min)

ASSESSMENT TECHNIQUES:
• quiz: Active recall testing (2-4 min)
• self_test: Personal knowledge checking (2-3 min)
• peer_review: Explain to others or imagine teaching (2-4 min)

RECALL TECHNIQUES:
• blurting: Free recall without prompts (2-4 min)
• free_recall: Unstructured memory retrieval (2-3 min)
• active_recall: Structured memory testing (3-4 min)

EFFECTIVE TECHNIQUE SEQUENCES:
1. For memorization-heavy content: flashcards → spaced_repetition → quiz
2. For conceptual content: feynman → analogies → self_test
3. For application content: practice_problems → case_studies → quiz
4. For mixed content: flashcards → feynman → quiz
5. Always end with assessment (quiz, self_test, or peer_review)

REQUIREMENTS:
- Each block covers ONE distinct unit/topic
- Choose 2-3 complementary techniques per block that build on each other
- Sequence techniques logically (input → processing → assessment)
- No duplicate techniques within the same block
- Techniques can repeat across different blocks
- Match technique types to content types (memorization, conceptual, application)
- Always include one assessment technique as the final technique in each block

CONTENT REQUIREMENTS FOR EACH TECHNIQUE:
Each technique description must specify:
1. What specific content to focus on with this technique
2. How to execute the technique effectively
3. What outcome to achieve before moving to next technique
4. Time allocation within the technique duration

CRITICAL: You MUST return a complete JSON object with ALL required fields.

REQUIRED JSON STRUCTURE - Return ONLY this JSON format:
{{
  "units_to_cover": ["Unit 1 Name", "Unit 2 Name", "Unit 3 Name"],
  "pomodoro": "25/5",
  "techniques": ["technique1", "technique2", "technique3"],
  "blocks": [
    {{
      "unit": "Unit 1 Name",
      "content_type": "memorization_heavy",
      "techniques": [
        {{
          "name": "flashcards",
          "sequence": 1,
          "duration": 4,
          "description": "Create flashcards for key terms and formulas. Focus on active recall of definitions and equations."
        }},
        {{
          "name": "spaced_repetition",
          "sequence": 2,
          "duration": 3,
          "description": "Review previously created flashcards with increasing intervals. Focus on difficult concepts."
        }},
        {{
          "name": "quiz",
          "sequence": 3,
          "duration": 5,
          "description": "Test knowledge with self-generated questions. Identify gaps for further review."
        }}
      ],
      "duration": {block_duration},
      "description": "Complete detailed description of the unit content covering all relevant subtopics."
    }}
  ]
}}

EXAMPLE COMPLETE RESPONSE:
{{
  "units_to_cover": ["Photosynthesis Overview", "Light Reactions", "Calvin Cycle"],
  "pomodoro": "25/5",
  "techniques": ["feynman", "flashcards", "quiz", "analogies", "practice_problems", "self_test"],
  "blocks": [
    {{
      "unit": "Photosynthesis Overview",
      "content_type": "conceptual_heavy",
      "techniques": [
        {{
          "name": "feynman",
          "sequence": 1,
          "duration": 5,
          "description": "Explain photosynthesis process in simple terms as if teaching a child. Focus on the overall equation and energy transformation."
        }},
        {{
          "name": "analogies",
          "sequence": 2,
          "duration": 3,
          "description": "Create analogies comparing photosynthesis to familiar processes like cooking or factory production."
        }},
        {{
          "name": "quiz",
          "sequence": 3,
          "duration": 4,
          "description": "Test understanding with questions about the overall process, inputs, outputs, and energy flow."
        }}
      ],
      "duration": 12,
      "description": "Photosynthesis converts light energy into chemical energy through two interconnected stages. Master equation: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Key concepts include energy transformation, chloroplast structure, and the relationship between light and dark reactions."
    }}
  ]
}}

Remember: Each block must have 2-3 techniques in logical sequence, with assessment as the final technique. Total duration per block is {block_duration} minutes."""
    
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
    """Generate study plan with GPT with enhanced validation for multi-technique blocks"""
    
    for attempt in range(max_retries + 1):
        try:
            print(f"🤖 GPT attempt {attempt + 1}/{max_retries + 1}")
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert curriculum designer. You MUST return ONLY valid JSON with ALL required fields and proper technique sequences. Each block must have 2-3 techniques with logical progression."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=3000,
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
            
            # Validate ALL required fields
            required_fields = ["blocks", "units_to_cover", "techniques", "pomodoro"]
            missing_fields = [field for field in required_fields if field not in parsed]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                raise ValueError(f"Missing required fields: {missing_fields}")
            
            blocks = parsed.get("blocks", [])
            if not blocks:
                raise ValueError("No blocks generated")
            
            # Validate block structure with techniques
            for i, block in enumerate(blocks):
                required_block_fields = ["unit", "techniques", "description", "duration"]
                missing_block_fields = [field for field in required_block_fields if field not in block]
                if missing_block_fields:
                    raise ValueError(f"Block {i} missing fields: {missing_block_fields}")
                
                # Validate techniques array
                techniques = block.get("techniques", [])
                if not techniques or len(techniques) < 2 or len(techniques) > 3:
                    raise ValueError(f"Block {i} must have 2-3 techniques, got {len(techniques)}")
                
                # Validate each technique
                for j, technique in enumerate(techniques):
                    required_technique_fields = ["name", "sequence", "duration", "description"]
                    missing_technique_fields = [field for field in required_technique_fields if field not in technique]
                    if missing_technique_fields:
                        raise ValueError(f"Block {i}, technique {j} missing fields: {missing_technique_fields}")
            
            print(f"✅ GPT generated valid multi-technique response with {len(blocks)} blocks")
            print(f"📊 Total techniques across all blocks: {sum(len(block['techniques']) for block in blocks)}")
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
    """Generate comprehensive study plan with multi-technique blocks"""
    
    try:
        user_id = extract_user_id(request)
        
        # Build enhanced prompt
        prompt = build_enhanced_prompt(data.objective, data.parsed_summary, data.duration)
        
        # Generate plan with GPT
        parsed = generate_gpt_plan(prompt)
        
        # Extract plan components
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        units = parsed.get("units_to_cover", [])
        all_techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")
        
        # Build study blocks with multiple techniques
        blocks = []
        context_tasks = []
        total_time = 0
        
        for idx, item in enumerate(blocks_json):
            unit = item.get("unit", f"Unit {idx + 1}")
            description = item.get("description", "Study the assigned material")
            duration = item.get("duration", 12)
            techniques_data = item.get("techniques", [])
            block_id = f"block_{uuid.uuid4().hex[:8]}"
            
            # Process techniques
            study_techniques = []
            primary_technique = None
            
            for tech_data in techniques_data:
                technique = StudyTechnique(
                    name=tech_data.get("name", "feynman"),
                    sequence=tech_data.get("sequence", 1),
                    duration=tech_data.get("duration", 4),
                    description=tech_data.get("description", "Apply this technique to the material")
                )
                study_techniques.append(technique)
                
                # First technique is primary
                if primary_technique is None:
                    primary_technique = technique.name
            
            # Sort techniques by sequence
            study_techniques.sort(key=lambda x: x.sequence)
            
            # Create study block
            study_block = StudyBlock(
                id=block_id,
                unit=unit,
                techniques=study_techniques,
                phase=primary_technique or "feynman",
                tool=primary_technique or "feynman",
                lovable_component="multi-technique-block",
                duration=duration,
                description=description,
                position=idx
            )
            
            blocks.append(study_block)
            total_time += duration
            
            technique_names = [t.name for t in study_techniques]
            print(f"📋 Block {idx + 1}: {unit}")
            print(f"   Techniques: {' → '.join(technique_names)} ({duration}min)")
            print(f"   Description: {description[:100]}...")
            
            # Prepare context updates for each technique
            for technique in study_techniques:
                context_payload = {
                    "source": "session_planner",
                    "user_id": user_id,
                    "current_topic": f"{unit} — {technique.name}",
                    "learning_event": {
                        "concept": unit,
                        "phase": technique.name,
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Planned {technique.name} session (seq {technique.sequence}): {technique.description[:200]}...",
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
        
        # Send final synthesis trigger
        if successful_updates > 0:
            try:
                final_payload = {
                    "source": "session_planner",
                    "user_id": user_id,
                    "current_topic": "Complete Multi-Technique Study Plan",
                    "learning_event": {
                        "concept": data.objective or "Generated Study Plan",
                        "phase": "planning",
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Comprehensive study plan with {len(blocks)} blocks, {sum(len(block.techniques) for block in blocks)} total techniques covering: {', '.join(units[:3])}{'...' if len(units) > 3 else ''}",
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
            techniques=all_techniques,
            blocks=blocks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 Study plan generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate study plan: {str(e)}")
