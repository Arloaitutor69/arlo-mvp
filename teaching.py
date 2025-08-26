from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
import os
import re

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

# --- GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor who excels in teaching difficult content in a way that is engaging and the most simple easy to understand way possible.
Create exactly 8-14 teaching blocks that thoroughly cover ALL aspects of the requested topic. Your explanations should sound like you’re talking directly to the student, never like a textbook. 

CRITICAL STYLE REQUIREMENTS:
- Always use **simple words** and explain technical terms in plain English the first time they appear.
- Always include **relatable analogies, examples, or metaphors**
- Always keep a **conversational tone**: ask rhetorical questions, say “think of it like…” or “imagine…”.
- Never drift into formal research paper or lecture style.
- Never introduce advanced words without breaking them down.
- Never output bullet lists without adding a quick analogy or everyday example to ground them.
- mimick exactly the assistant examples, particularly the casual easy to understand nature of explenations with lots of examples and clarifications. 

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas.

TEACHING BLOCK STRUCTURE:
- Each block should fully explain 1-2 subtopics in an easy to understand way.
- Cover all aspects of the requested topic comprehensively.
- Use bullet points with * for key concepts and lists.
- Use **bold formatting** for important terms and concepts.
- Include examples in parentheses when helpful.

CONTENT QUALITY STANDARDS:
- Each block should be ~70-130 words.
- ONLY MENTION information relevant to a test, not tangential information.
- Define all technical terms at first mention and assume student has almost zero prior knowledge

--- Most Important ---
1. Always output exactly 8-14 separate teaching blocks.
2. Mimic teaching style of examples as closely as possible, use same casual language, structure, and explanation style.
"""

# --- JSON examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "lesson": [
    {
      "title": "What is Economics?",
      "content": "Economics is the study of how people make choices about their limited resources. Everyone—individuals, businesses, and governments—has to make decisions about what to use, what to save, and what to trade.\\n\\n**Key ideas:**\\n* **Scarcity:** Resources (money, time, food, etc.) are limited. We can\\'t have everything we want.\\n* **Choices:** Because of scarcity, we make decisions about what to use resources for.\\n* **Opportunity Cost:** Whenever you choose one thing, you give up the next best alternative. (Example: if you spend $10 on lunch, you can\\'t spend that $10 on a movie ticket.)\\n\\nSo economics is the study of **who gets what, how they can get it, and why!**"
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "lesson": [
    {
      "title": "The Cell Membrane: Your Cell's Security System",
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\\n\\n**Key things to know:**\\n* It's made of a double layer of phospholipids (kind of like a thin soapy bubble wall)\\n* It is **selectively permeable** – a fancy term for deciding what goes in and what comes out\\n* It has special **transport proteins** that act like doors or ID checkers for bigger molecules when they want to enter or leave\\n\\n**What actually gets through:**\\n* Water and very small molecules can slip in and out easily\\n* Larger molecules need a special 'door' (transport proteins)\\n* Waste gets pushed out so the cell stays clean"
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "lesson": [
    {
      "title": "Micro vs. Macro Economics",
      "content": "Economics is split into two main 'worlds.'\\n\\n**Microeconomics:** The study of small, individual decisions.\\n* Example: A family choosing whether to eat out or cook at home\\n* Example: A business deciding how much to charge for sneakers\\n\\n**Macroeconomics:** The study of the whole economy.\\n* Example: Why is inflation rising?\\n* Example: Why do some countries grow richer while others struggle?\\n\\nThink of it like zooming in with a camera: **Micro = close-up**, **Macro = wide-angle** view of the entire economy."
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_4 = """
{
  "lesson": [
    {
      "title": "Cells and Cell Theory",
      "content": "A **cell** is the smallest living piece of life that can do all the important things like grow, use energy, react to surroundings, and reproduce.\\n\\n**Cell Theory says:**\\n* All living things are made of cells\\n* All cells come from other cells\\n\\n**Types of cells:**\\n* **Prokaryotes:** Simple cells with no nucleus, DNA floats in cytoplasm, reproduce fast by binary fission (split in two)\\n* **Eukaryotes:** Found in plants and animals, more complex with a nucleus to protect DNA, like miniature cities with factories and workers\\n\\nCells often team up to make bigger organisms (like humans with trillions of cells working together)."
    }
  ]
}
"""

# --- Helper utilities --- #
def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _block_valid(block: TeachingBlock) -> (bool, Optional[str]):
    if not isinstance(block.title, str) or not block.title.strip():
        return False, "missing or invalid title"
    if not isinstance(block.content, str) or not block.content.strip():
        return False, "missing or invalid content"
    words = _count_words(block.content)
    if words < 40:
        return False, f"content too short ({words} words)"
    if len(block.title) > 200:
        return False, "title too long"
    return True, None

def _sanitize_content(raw: str) -> str:
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    return s

def _validate_and_sanitize_blocks(blocks: List[TeachingBlock]) -> (bool, Optional[str], List[TeachingBlock]):
    sanitized = []
    for i, b in enumerate(blocks):
        if not isinstance(b.title, str) or not isinstance(b.content, str):
            return False, f"block {i} has invalid types", blocks
        temp_block = TeachingBlock(title=b.title, content=b.content)
        ok, reason = _block_valid(temp_block)
        if not ok:
            return False, f"block {i} invalid: {reason}", blocks
        sanitized_content = _sanitize_content(b.content)
        sanitized.append(TeachingBlock(title=b.title, content=sanitized_content))
    return True, None, sanitized

# --- OpenAI call wrapper --- #
def _call_model_and_get_parsed(input_messages, max_tokens=4000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=TeachingResponse,
        reasoning={"effort": "low"},
        instructions="Teach the topic in a fun, casual, and conversational style that mimics how a friendly tutor would explain things. Every block must include at least one analogy or real-life example, define any tricky term in plain language, and avoid textbook-like tone.",
        max_output_tokens=max_tokens,
    )

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        # Context info
        context_parts = []
        if req.subject:
            context_parts.append(f"Subject: {req.subject}")
        if req.level:
            context_parts.append(f"Level: {req.level}")
        if req.test_type:
            context_parts.append(f"Test: {req.test_type}")
        context_info = "\n".join(context_parts)

        # User prompt
        user_prompt = f"""{context_info}

Create a comprehensive lesson based on this study plan: {req.description}

Ensure every topic in the study plan is properly explained, and avoid veering from the study plan.
Output exactly 8-14 teaching blocks in valid JSON format with proper formatting including bullet points and bold text.
"""

        # Messages
        input_messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_4},
            {"role": "user", "content": user_prompt},
        ]


        # First attempt
        response = _call_model_and_get_parsed(input_messages)

        if getattr(response, "output_parsed", None) is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
            }
            response = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        lesson_blocks = response.output_parsed.lesson

        # Ensure 8–14 blocks
        if not (8 <= len(lesson_blocks) <= 14):
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: Must have 8-14 blocks. Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Lesson block count invalid ({len(lesson_blocks)}). Retry failed.")
            lesson_blocks = response_retry.output_parsed.lesson
            if not (8 <= len(lesson_blocks) <= 14):
                raise HTTPException(status_code=500, detail=f"Lesson block count invalid after retry ({len(lesson_blocks)}).")

        # Validate + sanitize
        valid, reason, sanitized_blocks = _validate_and_sanitize_blocks(lesson_blocks)
        if not valid:
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: Last output failed validation ({reason}). Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry failed.")
            lesson_blocks = response_retry.output_parsed.lesson
            valid2, reason2, sanitized_blocks2 = _validate_and_sanitize_blocks(lesson_blocks)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_blocks = sanitized_blocks2

        return TeachingResponse(lesson=sanitized_blocks)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating teaching content: {str(e)}")
