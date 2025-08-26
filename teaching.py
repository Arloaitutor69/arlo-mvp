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

# --- GPT System Prompt with JSON examples --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor creating who excels in teaching difficult content in the most simple easy to understand way possible.
Create exactly 8-14 teaching blocks that thoroughly cover ALL aspects of the requested topic.

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
- Each block should be ~50-130 words of teaching content (minimum lowered slightly).
- ONLY MENTION information relevant to a test, not tangential information.
- Explain concepts in extremely easy-to-understand, casual language.
- Use analogies, mnemonic devices, and other learning strategies when helpful.
- Define all technical terms at first mention.

--- Most Important ---
1. Always output exactly 8-14 separate teaching blocks. Treat each subtopic as its own block. Follow the formatting style of examples exactly with proper bullet points, bold text, and clear structure.
2. Mimic teaching style of examples as closely as possible, use same casual language, structure, and explanation style.
3. If you cannot follow the formatting rules exactly, return a single JSON object like: { "error": "short reason why rules couldn't be followed" } and nothing else.
"""

# --- Assistant few-shot examples (TRUE JSON examples for formatting/escaping guidance) --- #
# --- Assistant few-shot examples (TRUE JSON examples for formatting/escaping guidance) --- #
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

# --- Helper utilities for validation & sanitization --- #
def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _block_valid(block: TeachingBlock) -> (bool, Optional[str]):
    """Validate a single block for basic rules. Returns (is_valid, reason_if_invalid)."""
    if not isinstance(block.title, str) or not block.title.strip():
        return False, "missing or invalid title"
    if not isinstance(block.content, str) or not block.content.strip():
        return False, "missing or invalid content"
    # Require at least one bullet marker "* "
    if "* " not in block.content:
        return False, "no bullet list found (require '* ' bullets)"
    # Require at least one newline (either actual newline or escaped)
    if ("\n" not in block.content) and ("\\n" not in block.content):
        return False, "no newline breaks found (require \\n for paragraph breaks)"
    # Word count roughly within range (allow some slack)
    words = _count_words(block.content)
    if words < 40:  # slightly relaxed lower bound to avoid false negatives
        return False, f"content too short ({words} words)"
    # title length small sanity
    if len(block.title) > 200:
        return False, "title too long"
    return True, None

def _sanitize_content(raw: str) -> str:
    """
    Programmatically sanitize content to ensure:
      - internal double quotes are escaped as \"
      - newline characters are represented as \\n (literal backslash + n)
      - ensure bullets '* ' are preserved
    We assume raw is a Python string (parsed); we will convert actual newlines to \\n
    and escape double quotes.
    """
    # First, convert actual newline characters to the two-character sequence \n
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    # Escape double quotes
    s = s.replace('"', '\\"')
    return s

def _validate_and_sanitize_blocks(blocks: List[TeachingBlock]) -> (bool, Optional[str], List[TeachingBlock]):
    """
    Validate parsed blocks. If valid, sanitize content for JSON-safe output and return (True, None, sanitized_blocks).
    If invalid, return (False, reason, original_blocks).
    """
    sanitized = []
    for i, b in enumerate(blocks):
        # Validate structure types first
        if not isinstance(b.title, str) or not isinstance(b.content, str):
            return False, f"block {i} has invalid types", blocks
        # Create a temporary TeachingBlock for validation purposes (use raw strings)
        temp_block = TeachingBlock(title=b.title, content=b.content)
        ok, reason = _block_valid(temp_block)
        if not ok:
            return False, f"block {i} invalid: {reason}", blocks
        # Sanitize content (escape quotes and convert newlines to \\n)
        sanitized_content = _sanitize_content(b.content)
        sanitized.append(TeachingBlock(title=b.title, content=sanitized_content))
    return True, None, sanitized

# --- OpenAI call + retry logic encapsulated --- #
def _call_model_and_get_parsed(input_messages, max_tokens=4000, temperature=0.25):
    """
    Wrapper to call the Responses API via client.responses.parse and return the parsed output.
    """
    response = client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=TeachingResponse,
        reasoning={"effort": "low"},
        temperature=temperature,
        max_output_tokens=max_tokens
    )
    return response

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
Output exactly 8-14 teaching blocks in valid JSON format with proper formatting including bullet points and bold text.
"""

        # Prepare input messages including two TRUE JSON assistant examples
        input_messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
            {"role": "user", "content": user_prompt}
        ]

        # First attempt
        response = _call_model_and_get_parsed(input_messages, temperature=0.25)

        # If model didn't return parsed output, handle refusal/errors
        if getattr(response, "output_parsed", None) is None:
            # If refusal present, bubble up
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            # Otherwise attempt a single retry with a strict "Fix JSON only" instruction
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object that matches the TeachingResponse schema. Do not add commentary."
            }
            input_messages_retry = input_messages + [retry_msg]
            response = _call_model_and_get_parsed(input_messages_retry, temperature=0.0)
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        # Extract parsed lesson blocks
        lesson_blocks = response.output_parsed.lesson

        # Basic sanity check: ensure length is within 8-14
        if not (8 <= len(lesson_blocks) <= 14):
            # Try one automated retry instructing the model to return corrected JSON only
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: The last output did not have the correct number of blocks (must be 8-14). Return only a corrected JSON object that matches the TeachingResponse schema and obeys all prior formatting rules. Nothing else."
            }
            input_messages_retry = input_messages + [retry_msg]
            response_retry = _call_model_and_get_parsed(input_messages_retry, temperature=0.0)
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Lesson block count ({len(lesson_blocks)}) not within 8–14 range and retry failed.")
            lesson_blocks = response_retry.output_parsed.lesson
            # Re-check counts
            if not (8 <= len(lesson_blocks) <= 14):
                raise HTTPException(status_code=500, detail=f"Lesson block count ({len(lesson_blocks)}) not within 8–14 range after retry.")

        # Validate & sanitize blocks (programmatic sanitization ensures proper escaping and \\n usage)
        valid, reason, sanitized_blocks = _validate_and_sanitize_blocks(lesson_blocks)
        if not valid:
            # Retry once with a strict Fix JSON only prompt
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: The previous output failed validation ({reason}). Return only a corrected JSON object that matches the TeachingResponse schema and obeys all formatting rules. Nothing else."
            }
            input_messages_retry = input_messages + [retry_msg]
            response_retry = _call_model_and_get_parsed(input_messages_retry, temperature=0.0)
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry did not return valid parsed output.")
            lesson_blocks = response_retry.output_parsed.lesson
            valid2, reason2, sanitized_blocks2 = _validate_and_sanitize_blocks(lesson_blocks)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_blocks = sanitized_blocks2

        # At this point sanitized_blocks contains TeachingBlock objects with content sanitized for JSON (internal quotes escaped and newlines as \\n)
        # Return final TeachingResponse (pydantic will validate)
        return TeachingResponse(lesson=sanitized_blocks)

    except HTTPException:
        # re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating teaching content: {str(e)}"
        )
