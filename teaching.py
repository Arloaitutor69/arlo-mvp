from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os

# --- Initialize OpenAI client --- #
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

# --- Input schema --- #
class TeachingRequest(BaseModel):
    description: str
    subject: Optional[str] = None
    level: Optional[str] = None
    test_type: Optional[str] = None

# --- Output schema --- #
class TeachingBlock(BaseModel):
    title: str
    content: str

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- GPT System Prompt with examples --- #
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
Always output exactly 10-14 separate teaching blocks. Treat each subtopic as its own block. Follow the style of examples exactly.
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
Output exactly 10-14 teaching blocks in valid JSON format."""

        # --- OpenAI Responses API call using parse() --- #
        response = client.responses.parse(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            text_format=TeachingResponse,
            reasoning={"effort": "medium"},
            max_output_tokens=7000
        )

        # Handle refusals
        if hasattr(response, "refusal") and response.refusal:
            raise HTTPException(status_code=400, detail=response.refusal)

        # Extract lesson blocks
        lesson_blocks = response.output_parsed.lesson

        # Sanity check: ensure 10-14 blocks
        if not (10 <= len(lesson_blocks) <= 14):
            raise HTTPException(
                status_code=500,
                detail=f"Lesson block count ({len(lesson_blocks)}) not within 10–14 range"
            )

        return TeachingResponse(lesson=lesson_blocks)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating teaching content: {str(e)}"
        )
