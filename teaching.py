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
    type: Literal["overview", "mini_lesson", "summary"]
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
                "minItems": 9,
                "maxItems": 14,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "overview", 
                                "mini_lesson", 
                                "summary"
                            ]
                        },
                        "title": {
                            "type": "string",
                            "minLength": 1
                        },
                        "content": {
                            "type": "string",
                            "minLength": 70
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
2. Output ONLY valid JSON format with proper escaping
3. Use double quotes, escape internal quotes as \\"
4. Use \\n for line breaks within content
5. No trailing commas

TEACHING BLOCK STRUCTURE:
- First Block: breifly introduce topic and give 3 main questions going to answer
- Middle Blocks: mini lessons which fully flesh out 1-2 subtopics in an easy to understand way, modeled off example mini lessons
- Final Block: summary of what was learned

CONTENT QUALITY STANDARDS:
- each mini lesson should be 75-100 words of teaching content 
- ONLY MENTION information relevant to a test, not tangential information
- Explain concepts in extremely easy-to-understand, casual language
- Use analogies, mneumonic devices, and other learning stradegies when helpful
- Define all technical terms at first mention

EXAMPLE FORMAT - Economics

{
  "lesson": [
    {
      "type": "overview",
      "title": "What Are We Going to Learn About Economics?",
      "content": "**Main Questions:**\n\n• What is economics and why does it matter in daily life?\n• How do scarcity and choice affect everyone's decisions?\n• What's the difference between microeconomics and macroeconomics?\n\n**Learning Goals:**\n\n• Understand what economics is really about\n• Learn key concepts like scarcity and opportunity cost\n• See how economics affects both individual choices and whole countries\n• Compare micro and macro perspectives in simple terms"
    },
    {
      "type": "mini_lesson",
      "title": "What Is Economics, Really?",
      "content": "**Economics** is the study of how people make choices about their **limited resources**. Everyone—individuals, businesses, and governments—has to make decisions about what to use, what to save, and what to trade.\n\n**Key ideas:**\n• **Scarcity:** Resources (money, time, food, etc.) are **limited**. We can't have everything we want.\n• **Choices:** Because of scarcity, we make decisions about what to use resources for.\n• **Opportunity Cost:** Whenever you choose one thing, you give up the **next best alternative**. (Example: if you spend $10 on lunch, you can't spend that $10 on a movie ticket.)\n\nSo economics is the study of **who gets what, how they can get it, and why!**"
    },
    {
      "type": "mini_lesson",
      "title": "Micro vs. Macro Economics",
      "content": "Economics is split into two main \"worlds.\"\n\n• **Microeconomics:** The study of small, individual decisions.\n   • Example: A family choosing whether to eat out or cook at home.\n   • Example: A business deciding how much to charge for sneakers.\n\n• **Macroeconomics:** The study of the whole economy.\n   • Example: Why is inflation rising?\n   • Example: Why do some countries grow richer while others struggle?\n\nThink of it like zooming in with a camera: **Micro = zoomed-in close-up**, **Macro = wide angle view of the entire economy.**"
    }
  ]
}

here is a second cell biology example 

{
  "lesson": [
    {
      "type": "mini_lesson",
      "title": "What Is a Cell, Really?",
      "content": "A **cell** is the smallest living piece of life that can do all the important things like grow, use energy, react to its surroundings, and replicate to make new cells. **Cell theory** says that…\n• All living things are made of cells\n• All cells that you see came from another cell at one point\n\nThe most basic type of cell is called a **prokaryote**. These guys have only ONE cell, hence their name, single cell life forms. Prokaryotes are special in a few ways:\n• No nucleus (remember that big circle in the middle of the cell)\n• Their DNA floats freely in the cytoplasm (jelly like substance that fills the whole cell)\n• They reproduce extremely quickly by a process called binary fission (one cell that splits into two)\n\nThe cells that you have are called **Eukaryotes**—they are cells found in plants and animals, and are like miniature cities with their own factories, workers, and laws."
    },
    {
      "type": "mini_lesson",
      "title": "The Cell Membrane: Your Cell's Security System",
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\n\n**Key things to know:**\n• It's made of a double layer of phospholipids (kind of like a thin soapy bubble wall)\n• It is **selectively permeable** – a fancy term for deciding what goes in and what comes out\n• It has special transport proteins that act like doors or ID checkers for bigger molecules when they want to enter or leave\n\n**What actually gets through:**\n• Water and very small molecules can slip in and out easily\n• Larger molecules need a special 'door' (transport proteins)\n• Waste gets pushed out so the cell stays clean\n\nSome cells have a secondary wall that surrounds the cell membrane. The **Cell wall** is…\n• Found in **plant cells**, fungi, and bacteria — but **not in animal cells**\n• Made mostly of cellulose (a tough, sturdy substance)\n• Strong and rigid — it helps plants stand tall instead of flopping over"
    }
  ]
}


note: please mimick teaching style and content structure of examples, and try to have each lesson fully teach at least one subtopic. 
BLOCK TYPES TO USE:
- "overview": Introduction, learning goals, main questions
- "mini_lesson": In-depth explanations of concepts
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

Create a comprehensive lesson based on this study plan: {req.description}

ensure every topic in the study plan is properly explained, and avoid veering from the study plan. 

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
            reasoning_effort="low",
            max_completion_tokens=5000
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
                type=block.get("type", "mini_lesson"),
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
