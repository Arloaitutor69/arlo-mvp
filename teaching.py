# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union, Optional
import openai
import os
import json
import re

# Load OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

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
    content: Union[str, List[str]]

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- Enhanced GPT Prompt with Examples --- #
GPT_SYSTEM_PROMPT = """
You are an expert tutor creating comprehensive engaging easy to understand learning content. Your goal is to create exactly 8-12 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Create EXACTLY 10-14 teaching blocks - no more, no less
4. Focus ONLY on teaching content - no metadata, tips, or study time estimates

TEACHING BLOCK STRUCTURE:
- Block 1: overview of what student is going to learn, with main questions that will be answered etc
- Last Block: summary of what was learned

CONTENT QUALITY STANDARDS:
- MOST IMPORTANT: Explain concepts in extremely easy to understand and casual langauge, ensure it is understandable and engaging but dont waste words
- Use clear scaffolding and connect to prior knowledge
- Use evidence-based cognitive strategies such as analogies, metaphors, chunking ONLY when helpful and appropriate
- Use clear, student-friendly language while maintaining accuracy
- Define all technical terms at first mention and reinforce understanding through examples

FORMATTING EXAMPLES - Follow these patterns:

EXAMPLE OVERVIEW BLOCK 1:
{
  "type": "overview",
  "title": "What Are We Going to Learn?",
  "content": "**Main Questions:**\\n\\n* What is a cell?\\n* What are its key parts?\\n* How do cells divide and why does that matter?\\n\\n**By the end of this session, the learner should be able to:**\\n\\n* Explain what a eukaryotic cell is and what's inside it\\n* Describe the major organelles and their functions\\n* Understand the steps of mitosis and its importance"
}

EXAMPLE BLOCK 2:
{
  "type": "teaching",
  "title": "What Is a Cell, Really?",
  "content": "A **cell** is the smallest unit of life that can perform all life processes: growth, energy use, reproduction, and response to the environment.\\n\\n* Some organisms, like bacteria, are made of just **one** cell\\n* Others, like humans, are made of **trillions**, all working together\\n* Every cell comes from a **pre-existing cell** - a core idea called the **cell theory**\\n\\nThere are two main categories of cells:\\n\\n* **Prokaryotic cells** - Simple, small, and lack a nucleus (example: bacteria)\\n* **Eukaryotic cells** - Larger and more complex, with a defined nucleus and internal compartments (examples: human, plant, and fungal cells)\\n\\n**Analogy:** Think of a eukaryotic cell as a **tiny, self-sustaining city**, where each part of the city (called an organelle) has a specific job - from managing energy to protecting the borders."
}

EXAMPLE BLOCK 3:
{
  "type": "teaching",
  "title": "What's the Difference Between Prokaryotic and Eukaryotic Cells?",
  "content": "**Key Question:** How do simpler cells like bacteria compare to the complex cells found in humans?\\n\\nLet's break it down:\\n\\n**Prokaryotic Cells:**\\n\\n* No nucleus - DNA floats freely in the cytoplasm\\n* Lack membrane-bound organelles\\n* Smaller in size, structurally simpler\\n* Example: Bacteria\\n\\n**Eukaryotic Cells:**\\n\\n* Contain a nucleus where DNA is stored\\n* Have membrane-bound organelles that carry out specific functions\\n* Larger and more organized internally\\n* Examples: Animal cells, plant cells, fungi, protists\\n\\n**Helpful mnemonic:** Pro = No (nucleus), Eu = True (nucleus)\\n\\nThis simple phrase reminds students that **prokaryotes** do **not** have a nucleus, but **eukaryotes** do.\\n\\nUnderstanding this distinction is critical: nearly all cells studied in introductory biology are **eukaryotic** - so from here on, we'll focus on them."
}

EXAMPLE Block 6:
{
  "type": "teaching",
  "title": "How Does a Cell Make and Move Proteins?",
  "content": "**Key Question:** How are proteins made inside a cell, and how do they reach their destination?\\n\\nProteins are essential to life - they make up muscles, enzymes, hormones, and more. The cell uses a coordinated network of organelles to produce, process, and transport them:\\n\\n1. **Ribosomes**\\n\\n   * These are the builders. They take instructions from the nucleus and link together amino acids to form proteins.\\n   * Ribosomes are either floating freely or attached to the rough ER.\\n\\n2. **Rough Endoplasmic Reticulum (Rough ER)**\\n\\n   * The rough ER is a folded membrane system dotted with ribosomes.\\n   * It helps fold and process proteins after they're made and prepares them for shipment.\\n\\n3. **Golgi Apparatus**\\n\\n   * Once proteins leave the ER, they head to the Golgi.\\n   * This organelle modifies the proteins, adds molecular tags, and ships them where they need to go - inside or outside the cell.\\n\\n**Analogy:**\\n\\n* Ribosomes = factory workers\\n* Rough ER = the production line\\n* Golgi apparatus = the packaging and shipping department\\n\\nIf any step in this chain is disrupted, the cell can't function properly - proteins won't be delivered where they're needed, leading to dysfunction and even disease."
}

KEY FORMATTING RULES:
- Use **bold** for emphasis and key terms
- Use bullet points (*) for lists
- Use numbered lists (1., 2., 3.) for sequential processes
- Include clear section breaks and visual hierarchy
- Maintain consistent indentation for nested bullet points

RESPONSE FORMAT (JSON only):
{
  "lesson": [
    {
      "type": "overview",
      "title": "Complete Overview of [Topic]",
      "content": "Follow the formatting examples above with **Main Questions:** and learning objectives..."
    },
    {
      "type": "key_concepts", 
      "title": "Fundamental Concepts",
      "content": "Use bullet points, bold key terms, and analogies as shown in examples..."
    },
    {
      "type": "detailed_explanation",
      "title": "Subtopic 1 Deep Dive",
      "content": "Follow the detailed explanation format with **Key Question:** and structured content..."
    }
  ]
}

IMPORTANT: 
- mimick teaching style of example content and Follow the exact formatting patterns from the examples
- make all words clear simple and easy to understand. Don't overcomplicate material or use unnecesary jargon 
- Output ONLY the JSON response - no additional text
- Ensure exactly 8-12 blocks total
- Each block must contain substantial and accurate educational content
- Use the same markdown formatting, bullet points, and structure as the examples
- ESCAPE ALL SPECIAL CHARACTERS: Use \\n for newlines, \\\" for quotes, \\\\ for backslashes
"""

def clean_json_string(text):
    """Clean and sanitize JSON string to prevent parsing errors"""
    # Remove any potential control characters that might break JSON parsing
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Only clean up actual newlines and carriage returns if they're not already escaped
    # Don't double-escape quotes - GPT should already handle this
    
    return text

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        # Enhanced context for better content generation
        context_info = ""
        if req.subject:
            context_info += f"Subject: {req.subject}\n"
        if req.level:
            context_info += f"Academic Level: {req.level}\n"
        if req.test_type:
            context_info += f"Test Type: {req.test_type}\n"
        
        user_prompt = f"""{context_info}
Topic to teach: {req.description}

CRITICAL: Ensure all special characters are properly escaped in the JSON response. Use \\n for newlines, \\\" for quotes, etc."""

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Updated API call for GPT-5 Nano
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            # Removed temperature parameter since gpt-5-nano only supports default (1)
        )

        raw_output = response["choices"][0]["message"]["content"]
        
        # Clean up response to ensure it's valid JSON
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3]
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3]
        
        # Only remove control characters, don't double-escape
        raw_output = clean_json_string(raw_output)
        
        # Try to parse JSON with better error handling
        try:
            parsed_output = json.loads(raw_output)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to fix common issues
            print(f"JSON parsing error: {e}")
            print(f"Raw output: {raw_output[:500]}...")  # Log first 500 chars for debugging
            
            # Try to fix by removing any remaining problematic characters
            raw_output = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', raw_output)
            
            # Try parsing again
            try:
                parsed_output = json.loads(raw_output)
            except json.JSONDecodeError:
                # If it still fails, create a fallback response
                raise HTTPException(
                    status_code=500, 
                    detail="Failed to parse AI response. The AI generated malformed JSON."
                )

        # Validate and clean lesson blocks
        lesson_blocks = parsed_output.get("lesson", [])
        
        # Ensure we have 8-12 blocks
        if len(lesson_blocks) < 8:
            raise HTTPException(status_code=500, detail="Insufficient teaching blocks generated")
        if len(lesson_blocks) > 12:
            lesson_blocks = lesson_blocks[:12]  # Trim to 12 blocks
        
        # Validate each block
        for i, block in enumerate(lesson_blocks):
            if not isinstance(block, dict):
                raise HTTPException(status_code=500, detail=f"Invalid block structure at index {i}")

            # Validate required fields
            if "type" not in block or block["type"] not in [
                "overview", "key_concepts", "detailed_explanation", "examples", 
                "test_strategies", "practice_questions", "memory_aids", "common_mistakes"
            ]:
                block["type"] = "detailed_explanation"

            if not block.get("title"):
                block["title"] = f"Teaching Block {i+1}"

            # Clean up content - preserve formatting for all content types
            content = block.get("content", "")
            
            # Handle list content for key_concepts and similar types
            if block["type"] in ["key_concepts", "practice_questions", "memory_aids"]:
                if not isinstance(content, list):
                    # Keep as string to preserve formatting
                    if isinstance(content, str):
                        block["content"] = content
                    else:
                        block["content"] = str(content)
                else:
                    # Convert list back to formatted string
                    block["content"] = "\n\n".join([str(item) for item in content])
            else:
                # Ensure string content for other types
                if isinstance(content, list):
                    block["content"] = "\n\n".join([str(item) for item in content])
                else:
                    block["content"] = str(content)

        # Return only the lesson blocks
        return {"lesson": lesson_blocks}

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating teaching content: {str(e)}")
