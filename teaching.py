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

# --- GPT System Prompt with JSON examples --- #
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
- Use bullet points with * for key concepts and lists
- Use **bold formatting** for important terms and concepts
- Include examples in parentheses when helpful

CONTENT QUALITY STANDARDS:
- Each block should be 75-150 words of teaching content
- ONLY MENTION information relevant to a test, not tangential information
- Explain concepts in extremely easy-to-understand, casual language
- Use analogies, mnemonic devices, and other learning strategies when helpful
- Define all technical terms at first mention
- Include learning goals and key takeaways where appropriate

EXAMPLE TEACHING CONTENT (JSON FORMAT):

{
  "lesson": [
    {
      "title": "What is Economics?",
      "content": "Economics is the study of how people make choices about their limited resources. Everyone—individuals, businesses, and governments—has to make decisions about what to use, what to save, and what to trade.\\n\\n**Key ideas:**\\n* **Scarcity:** Resources (money, time, food, etc.) are limited. We can't have everything we want.\\n* **Choices:** Because of scarcity, we make decisions about what to use resources for.\\n* **Opportunity Cost:** Whenever you choose one thing, you give up the next best alternative. (Example: if you spend $10 on lunch, you can't spend that $10 on a movie ticket.)\\n\\nSo economics is the study of **who gets what, how they can get it, and why!**"
    },
    {
      "title": "Micro vs. Macro Economics",
      "content": "Economics is split into two main \\\"worlds.\\\"\\n\\n* **Microeconomics:** The study of small, individual decisions.\\n  * Example: A family choosing whether to eat out or cook at home.\\n  * Example: A business deciding how much to charge for sneakers.\\n* **Macroeconomics:** The study of the whole economy.\\n  * Example: Why is inflation rising?\\n  * Example: Why do some countries grow richer while others struggle?\\n\\nThink of it like zooming in with a camera: **Micro = zoomed-in close-up, Macro = wide angle view of the entire economy.**"
    },
    {
      "title": "What is a Cell and Why Does It Matter?",
      "content": "**Cells are the basic building block of all life.** There is a lot to learn about them, but we are just going to focus on the most important information you need to know.\\n\\n**The Main Questions we are going to tackle are:**\\n* What is a cell and why do people call it the basic unit of life?\\n* What are the main parts inside a cell and what are their jobs?\\n* How are plant and animal cells the same, and how are they different?\\n\\n**Learning Goals:**\\n* Get a simple understanding of what cells are and why they matter\\n* Learn the main jobs of the organelles (cell parts)\\n* See how the parts of a cell work together like a team"
    },
    {
      "title": "Cell Types: Prokaryotes vs. Eukaryotes",
      "content": "A **cell** is the smallest living piece of life that can do all the important things like grow, use energy, react to its surroundings, and replicate to make new cells. **Cell theory** says that…\\n* All living things are made of cells\\n* All cells that you see came from another cell at one point\\n\\nThe most basic type of cell is called a **prokaryote.** These guys have only ONE cell, hence they're name, single cell life forms\\n**Prokaryote's are special in a few ways:**\\n* No nucleus (remember that big circle in the middle of the cell)\\n* Their DNA floats freely in the cytoplasm (jelly like substance that fills the whole cell)\\n* They reproduce extremely quickly by a process called **binary fission** (one cell that splits into two)\\n\\nThe cells that you have are called **Eukaryotes**—they are cells found in plants and animals, and are like miniature cities with their own factories, workers, and laws."
    },
    {
      "title": "The Cell Membrane: Your Cell's Security System",
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\\n\\n**Key things to know:**\\n* It's made of a double layer of phospholipids (kind of like a thin soapy bubble wall)\\n* It is **selectively permeable** – a fancy term for deciding what goes in and what comes out\\n* It has special **transport proteins** that act like doors or ID checkers for bigger molecules when they want to enter or leave\\n\\n**What actually gets through:**\\n* Water and very small molecules can slip in and out easily\\n* Larger molecules need a special 'door' (transport proteins)\\n* Waste gets pushed out so the cell stays clean\\n\\nSome cells have a secondary wall that surrounds the cell membrane. The **Cell wall** is found in plant cells, fungi, and bacteria — but not in animal cells."
    }
  ]
}

--- IMPORTANT ADDITION ---
Always output exactly 10-14 separate teaching blocks. Treat each subtopic as its own block. Follow the formatting style of examples exactly with proper bullet points, bold text, and clear structure.
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
Output exactly 10-14 teaching blocks in valid JSON format with proper formatting including bullet points and bold text.
"""

        # --- OpenAI Responses API call using parse() --- #
        response = client.responses.parse(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": GPT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            text_format=TeachingResponse,
            reasoning={"effort": "low"},
            max_output_tokens=7000
        )

        # Handle refusals and parsing errors
        if response.output_parsed is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            else:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output.")

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
