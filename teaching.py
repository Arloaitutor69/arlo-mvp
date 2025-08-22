# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union, Optional
from openai import OpenAI
import os
import json
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

# --- Input schema: enhanced with context --- #
class TeachingRequest(BaseModel):
    description: str
    subject: Optional[str] = None  # e.g., "Biology", "History", "Mathematics"
    level: Optional[str] = None    # e.g., "High School", "College", "Graduate"
    test_type: Optional[str] = None # e.g., "SAT", "AP Exam", "Midterm", "Final"

# --- Simplified Output schema --- #
class TeachingBlock(BaseModel):
    type: Literal["overview", "key_concepts", "detailed_explanation", "examples", "test_strategies", "practice_questions", "memory_aids", "common_mistakes"]
    title: str
    content: str  # Always string to avoid confusion

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- Enhanced GPT Prompt with Improved Examples --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor creating comprehensive, engaging, easy-to-understand learning content. Create exactly 10 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Create EXACTLY 10-14 blocks
2. Output ONLY valid JSON format with proper escaping
3. Use double quotes, escape internal quotes as \\"
4. Use \\n for line breaks within content
5. No trailing commas

TEACHING BLOCK STRUCTURE:
- Block 1: overview of what student will learn, main questions
- last block: summary/memory aids of what was learned

CONTENT QUALITY STANDARDS:
- Explain concepts in extremely easy-to-understand, casual language
- Use clear scaffolding and connect to prior knowledge
- Use analogies and metaphors when helpful
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
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\\n\\n**Key things to know:**\\n• It’s made of a double layer of lipids (kind of like a thin soapy bubble wall)\\n• It is **selectively permeable** – only certain things are allowed through\\n• It has special proteins that act like doors or ID checkers for bigger molecules\\n\\n**What gets through:**\\n• Water and very small molecules can slip in and out easily\\n• Larger molecules need a special 'door' (transport proteins)\\n• Waste gets pushed out so the cell stays clean\\n\\n**Why it matters:** Without this guard, the cell could either dry up or burst like a water balloon."
    },
    {
      "type": "detailed_explanation",
      "title": "The Nucleus: Cell's Control Center",
      "content": "The **nucleus** is like the cell’s brain. It stores DNA and tells the cell what to do.\\n\\n**Main parts inside:**\\n• **Nuclear envelope** – a double wall that protects the DNA\\n• **Chromatin** – DNA wrapped around proteins, like thread on a spool\\n• **Nucleolus** – makes ribosomes (tiny machines that build proteins)\\n\\n**What the nucleus does:**\\n• Stores the genetic information (the instruction manual)\\n• Decides which instructions to follow at any given time\\n• Helps control protein building and cell reproduction\\n\\n**Fun fact:** The nucleus is only about 10% of the cell’s space but it holds almost all of the cell’s DNA — like squeezing a whole library into one room!"
    }
  ]
}


Remember: Output ONLY valid JSON with exactly 10 blocks. Each block must contain substantial educational content with proper JSON escaping."""

def fix_json_escaping(text):
    """Fix common JSON escaping issues"""
    # Remove any control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Fix single quotes that should be escaped double quotes
    text = re.sub(r"(?<!\\)'", '"', text)
    
    # Fix unescaped quotes inside content strings
    # Look for content field and fix quotes within it
    def fix_content_quotes(match):
        content_part = match.group(1)
        # Escape any unescaped quotes inside the content
        content_part = re.sub(r'(?<!\\)"', r'\\"', content_part)
        return f'"content": "{content_part}"'
    
    # Apply the fix to content fields
    text = re.sub(r'"content":\s*"([^"]*(?:\\.[^"]*)*)"', fix_content_quotes, text)
    
    # Remove trailing commas before closing brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    return text

def create_fallback_response(description, subject=None):
    """Create a fallback response when JSON parsing fails"""
    context = f" in {subject}" if subject else ""
    
    return {
        "lesson": [
            {
                "type": "overview",
                "title": f"Learning About {description.title()}",
                "content": f"We'll explore {description}{context}. This lesson covers the key concepts, practical applications, and important details you need to understand."
            },
            {
                "type": "detailed_explanation",
                "title": "Core Concepts",
                "content": f"Let's break down {description} into its fundamental parts and understand how they work together."
            },
            {
                "type": "detailed_explanation", 
                "title": "Key Principles",
                "content": f"The main principles underlying {description} help us understand why and how these processes occur."
            },
            {
                "type": "examples",
                "title": "Real-World Examples",
                "content": f"Here are practical examples of {description} that you encounter in everyday life."
            },
            {
                "type": "detailed_explanation",
                "title": "Important Details",
                "content": f"These specific details about {description} are crucial for a complete understanding."
            },
            {
                "type": "detailed_explanation",
                "title": "How It Works",
                "content": f"The step-by-step process of how {description} functions in practice."
            },
            {
                "type": "common_mistakes",
                "title": "Common Misconceptions",
                "content": f"Students often misunderstand these aspects of {description}. Let's clarify them."
            },
            {
                "type": "test_strategies",
                "title": "Study Tips",
                "content": f"Effective strategies for learning and remembering {description}."
            },
            {
                "type": "practice_questions",
                "title": "Practice Problems",
                "content": f"Test your understanding with these questions about {description}."
            },
            {
                "type": "memory_aids",
                "title": "Key Takeaways",
                "content": f"Remember these essential points about {description} for long-term retention."
            }
        ]
    }

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        # Build context efficiently
        context_parts = []
        if req.subject:
            context_parts.append(f"Subject: {req.subject}")
        if req.level:
            context_parts.append(f"Level: {req.level}")
        if req.test_type:
            context_parts.append(f"Test: {req.test_type}")
        
        context_info = "\n".join(context_parts)
        
        user_prompt = f"""{context_info}

Create a comprehensive lesson about: {req.description}

Output valid JSON with exactly 10 teaching blocks."""

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # API call with speed optimization
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            reasoning_effort="low"
        )

        raw_output = response.choices[0].message.content.strip()
        
        # Remove code block markers
        if raw_output.startswith("```"):
            raw_output = re.sub(r'^```(?:json)?\n?', '', raw_output)
            raw_output = re.sub(r'\n?```$', '', raw_output)
        
        # Fix JSON escaping issues
        raw_output = fix_json_escaping(raw_output)
        
        # Parse JSON with better error handling
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw output sample: {raw_output[:200]}...")
            # Use fallback response
            parsed_output = create_fallback_response(req.description, req.subject)

        # Validate and clean lesson blocks
        lesson_blocks = parsed_output.get("lesson", [])
        
        # Ensure exactly 10 blocks
        if len(lesson_blocks) != 10:
            if len(lesson_blocks) < 10:
                # Pad with additional blocks if needed
                while len(lesson_blocks) < 10:
                    lesson_blocks.append({
                        "type": "detailed_explanation",
                        "title": f"Additional Concepts",
                        "content": f"Additional important information about {req.description}."
                    })
            else:
                # Trim to exactly 10
                lesson_blocks = lesson_blocks[:10]
        
        # Clean and validate each block
        valid_types = {
            "overview", "key_concepts", "detailed_explanation", "examples", 
            "test_strategies", "practice_questions", "memory_aids", "common_mistakes"
        }
        
        for i, block in enumerate(lesson_blocks):
            # Ensure block is a dictionary
            if not isinstance(block, dict):
                block = {"type": "detailed_explanation", "title": f"Concept {i+1}", "content": "Important information about the topic."}
                lesson_blocks[i] = block

            # Validate type
            if "type" not in block or block["type"] not in valid_types:
                block["type"] = "detailed_explanation"

            # Ensure title exists
            if not block.get("title"):
                block["title"] = f"Learning Block {i+1}"

            # Ensure content is string and clean it
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\\n\\n".join([str(item) for item in content])
            elif not isinstance(content, str):
                content = str(content)
            
            # Clean content string
            content = content.replace('\n', '\\n').replace('\r', '')
            block["content"] = content

        return {"lesson": lesson_blocks}

    except Exception as e:
        print(f"Error in generate_teaching_content: {str(e)}")
        # Return fallback response on any error
        fallback = create_fallback_response(req.description, req.subject)
        return fallback
