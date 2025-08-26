from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import json

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

# --- Input schema --- #
class TeachingRequest(BaseModel):
    description: str
    subject: Optional[str] = None  # e.g., "Biology", "History", "Mathematics"
    level: Optional[str] = None    # e.g., "High School", "College", "Graduate"
    test_type: Optional[str] = None # e.g., "SAT", "AP Exam", "Midterm", "Final"

# --- Output schema --- #
class TeachingBlock(BaseModel):
    title: str
    content: str

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- JSON Schema for structured outputs --- #
TEACHING_SCHEMA = {
    "name": "teaching_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "lesson": {
                "type": "array",
                "minItems": 10,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "content": {"type": "string", "minLength": 600}
                    },
                    "required": ["title", "content"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["lesson"],
        "additionalProperties": False
    }
}

# --- GPT System Prompt with original examples + instruction --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor creating comprehensive, engaging, easy-to-understand learning content. 
Create exactly 10-14 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping
3. Use double quotes, escape internal quotes as \\\"\"
4. Use \\n for line breaks within content
5. No trailing commas

TEACHING BLOCK STRUCTURE:
- Each block should fully explain 1-2 subtopics in an easy to understand way
- Cover all aspects of the requested topic comprehensively
- Progress logically from foundational concepts to more complex ideas

CONTENT QUALITY STANDARDS:
- Each block should be 75-100 words of teaching content 
- ONLY MENTION information relevant to a test, not tangential information
- Explain concepts in extremely easy-to-understand, casual language
- Use analogies, mnemonic devices, and other learning strategies when helpful
- Define all technical terms at first mention

EXAMPLE TEACHING CONTENT:

--- Economics Example ---
Title: What Is Economics, Really?
Content: Economics is the study of how people make choices about limited resources. Everyone—individuals, businesses, and governments—has to decide what to use, what to save, and what to trade. Key ideas: Scarcity = resources are limited. Choices = decisions on resource use. Opportunity Cost = value of the next best alternative given up. Example: Spending $10 on lunch means you cannot spend it on a movie ticket. Economics studies who gets what, how, and why.

Title: Micro vs. Macro Economics
Content: Economics is split into two main areas. Microeconomics studies small, individual decisions (a family choosing to cook at home or eat out, a business setting prices). Macroeconomics studies the whole economy (why inflation rises, why some countries grow richer). Think of it as zooming with a camera: micro = close-up, macro = wide-angle view of the economy.

--- Cell Biology Example ---
Title: What Is a Cell, Really?
Content: A cell is the smallest living unit that can grow, use energy, react to its surroundings, and replicate. Cell theory: all living things are made of cells, all cells come from other cells. Prokaryotes are single-celled, lack a nucleus, DNA floats freely, reproduce quickly via binary fission. Eukaryotes are more complex, found in plants and animals, like miniature cities with factories, workers, and rules.

Title: The Cell Membrane: Your Cell's Security System
Content: The cell membrane is like a bouncer at the door, deciding what enters and exits. Key points: made of a phospholipid bilayer, selectively permeable (controls passage of molecules), uses transport proteins for larger molecules. Water and small molecules pass easily; waste is expelled to keep the cell clean. Some cells have a secondary cell wall (plants, fungi, bacteria) made of cellulose, strong and rigid for structural support.

--- IMPORTANT ADDITION ---
Always output exactly 10-14 separate teaching blocks. Do not merge examples. Treat each subtopic as its own block. Follow the style of the examples exactly.
"""

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        # Build context information
        context_parts = []
        if req.subject:
            context_parts.append(f"Subject: {req.subject}")
        if req.level:
            context_parts.append(f"Level: {req.level}")
        if req.test_type:
            context_parts.append(f"Test: {req.test_type}")
        
        context_info = "\n".join(context_parts)
        
        # Create user prompt
        user_prompt = f"""{context_info}

Create a comprehensive lesson based on this study plan: {req.description}

Ensure every topic in the study plan is properly explained, and avoid veering from the study plan. 

Output valid JSON with exactly 10-14 teaching blocks."""

        # --- OpenAI Responses API call ---
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": TEACHING_SCHEMA
            },
            reasoning={"effort": "low"},
            max_output_tokens=5000
        )

        # Extract validated structured JSON
        raw_content = response.output[0].content[0].text
        parsed_data = json.loads(raw_content)
        
        # Debug logging
        print(f"Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
        print(f"Raw content sample: {raw_content[:200]}...")
        
        # Find lesson array
        lesson_data = parsed_data.get("lesson")
        if not lesson_data:
            raise HTTPException(
                status_code=500,
                detail=f"No lesson array found in response. Keys: {list(parsed_data.keys())}"
            )
        
        # Convert to Pydantic models
        lesson_blocks = [
            TeachingBlock(
                title=block.get("title", f"Learning Block {i+1}"),
                content=block.get("content", "Educational content")
            )
            for i, block in enumerate(lesson_data)
        ]
        
        return TeachingResponse(lesson=lesson_blocks)

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse response as JSON: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating teaching content: {str(e)}"
        )
