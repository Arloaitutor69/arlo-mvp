# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
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
    type: Literal["overview", "key_concepts", "detailed_explanation", "examples", "summary"]
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
                        "type": {
                            "type": "string",
                            "enum": [
                                "overview", 
                                "key_concepts", 
                                "detailed_explanation", 
                                "examples", 
                                "summary"
                            ]
                        },
                        "title": {
                            "type": "string",
                            "minLength": 1
                        },
                        "content": {
                            "type": "string",
                            "minLength": 50
                        }
                    },
                    "required": ["type", "title", "content"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["lesson"],
        "additionalProperties": False
    }
}

# --- Enhanced GPT Prompt with Improved Examples --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor creating comprehensive, engaging, easy-to-understand learning content. Create exactly 10-14 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Create EXACTLY 10-14 blocks
2. Output ONLY valid JSON format with proper escaping
3. Use double quotes, escape internal quotes as \\"
4. Use \\n for line breaks within content
5. No trailing commas

TEACHING BLOCK STRUCTURE:
- Block 1: overview of what student will learn, main questions
- last block: summary of what was learned

CONTENT QUALITY STANDARDS:
- ONLY MENTION information relevant to a class/test, not tangential information or outside applications
- vary up teaching stradegies and structure of each component of lesson
- Explain concepts in extremely easy-to-understand, casual language
- Use clear scaffolding and connect to prior knowledge
- Use analogies when helpful
- Define all technical terms at first mention
- Student-friendly language while maintaining accuracy

EXAMPLE FORMAT - Cell Biology Topic:
{
  "lesson": [
    {
      "type": "overview",
      "title": "What Are We Going to Learn About Cells?",
      "content": "**Main Questions:**\\n\\n• What is a cell and why do people call it the basic unit of life?\\n• What are the main parts inside a cell and what are their jobs?\\n• How are plant and animal cells the same, and how are they different?\\n\\n**Learning Goals:**\\n\\n• Get a simple understanding of what cells are and why they matter\\n• Learn the main jobs of the organelles (cell parts)\\n• See how the parts of a cell work together like a team\\n• Compare plant and animal cells in an easy way"
    },
    {
      "type": "detailed_explanation",
      "title": "What Is a Cell, Really?",
      "content": "A **cell** is the smallest living piece of life that can do all the important things like grow, use energy, react to its surroundings, and make new cells.\\n\\n• Some life forms, like bacteria, are made of just **one** cell that does everything itself\\n• Bigger organisms, like humans, are made of **trillions** of cells all working together\\n• Every new cell comes from another cell — this is the basic rule of cell theory\\n\\n**Two main kinds of cells:**\\n• **Prokaryotic cells** – simple cells with no nucleus (like bacteria)\\n• **Eukaryotic cells** – more complex, have a nucleus (like plants and animals)\\n\\n**Analogy:** A eukaryotic cell is like a **mini city**, with each part doing a specific job to keep the whole place running."
    },
    {
      "type": "detailed_explanation",
      "title": "The Cell Membrane: Your Cell's Security Guard",
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\\n\\n**Key things to know:**\\n• It's made of a double layer of lipids (kind of like a thin soapy bubble wall)\\n• It is **selectively permeable** – only certain things are allowed through\\n• It has special proteins that act like doors or ID checkers for bigger molecules\\n\\n**What gets through:**\\n• Water and very small molecules can slip in and out easily\\n• Larger molecules need a special 'door' (transport proteins)\\n• Waste gets pushed out so the cell stays clean\\n\\n**Why it matters:** Without this guard, the cell could either dry up or burst like a water balloon."
    },
    {
      "type": "detailed_explanation",
      "title": "The Nucleus: Cell's Control Center",
      "content": "The **nucleus** is like the cell's brain. It stores DNA and tells the cell what to do.\\n\\n**Main parts inside:**\\n• **Nuclear envelope** – a double wall that protects the DNA\\n• **Chromatin** – DNA wrapped around proteins, like thread on a spool\\n• **Nucleolus** – makes ribosomes (tiny machines that build proteins)\\n\\n**What the nucleus does:**\\n• Stores the genetic information (the instruction manual)\\n• Decides which instructions to follow at any given time\\n• Helps control protein building and cell reproduction\\n\\n**Fun fact:** The nucleus is only about 10% of the cell's space but it holds almost all of the cell's DNA — like squeezing a whole library into one room!"
    }
  ]
}

BLOCK TYPES TO USE:
- "overview": Introduction, learning goals, main questions
- "key_concepts": Essential terms and principles
- "detailed_explanation": In-depth explanations of concepts
- "examples": Real-world applications and illustrations
- "summary": Summary and key takeaways

Remember: Output ONLY valid JSON with exactly 10-14 blocks. Each block must contain substantial educational content with proper JSON escaping."""

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

Create a comprehensive lesson about: {req.description}

Output valid JSON with exactly 10-14 teaching blocks."""

        # Prepare messages
        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": TEACHING_SCHEMA
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        # Debug: print what we actually got
        print(f"Parsed data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
        print(f"Raw content sample: {raw_content[:200]}...")
        
        # Handle different possible response structures
        if "lesson" in parsed_data:
            lesson_data = parsed_data["lesson"]
        elif isinstance(parsed_data, list):
            # If the response is directly a list of blocks
            lesson_data = parsed_data
        else:
            # Fallback: look for any array in the response
            lesson_data = None
            for key, value in parsed_data.items():
                if isinstance(value, list):
                    lesson_data = value
                    break
            
            if lesson_data is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"No lesson array found in response. Keys: {list(parsed_data.keys())}"
                )
        
        # Convert to Pydantic models for additional validation
        lesson_blocks = [
            TeachingBlock(
                type=block.get("type", "detailed_explanation"),
                title=block.get("title", f"Learning Block {i+1}"),
                content=block.get("content", "Educational content")
            )
            for i, block in enumerate(lesson_data)
        ]
        
        return TeachingResponse(lesson=lesson_blocks)

    except json.JSONDecodeError as e:
        # This should never happen with structured outputs, but just in case
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse response as JSON: {str(e)}"
        )
    
    except Exception as e:
        # Handle any other errors
        raise HTTPException(
            status_code=500,
            detail=f"Error generating teaching content: {str(e)}"
        )
