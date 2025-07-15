# teaching.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Union, Optional
import openai
import os
import json

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

# --- Enhanced GPT Prompt --- #
GPT_SYSTEM_PROMPT = """
You are an expert tutor creating comprehensive learning content. Your goal is to create exactly 8-12 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Create EXACTLY 8-12 teaching blocks - no more, no less
2. Each block must be substantial (200-300 words of teaching content)
3. Cover 100% of the topic comprehensively - leave nothing out
4. Focus ONLY on teaching content - no metadata, tips, or study time estimates

TEACHING BLOCK STRUCTURE:
- Block 1: Always "overview" - comprehensive introduction to the entire topic
- Blocks 2-5: "key_concepts" - give introduction to key concepts, basic definitions and intro that will be unpacked later
- Blocks 6-10: "detailed_explanation" - unpack the concepts in comprehensive depth
- Blocks 11-12: "summary" - summarize all content with active recall, memory aids like mneumonics or memory palace

CONTENT QUALITY STANDARDS:
- Each detailed_explanation block should be 200-300 words
- Explain concepts as if teaching a complete lesson
- Include all relevant subtopics, processes, exceptions
- Use clear scaffolding and connect to prior knowledge
- Make content accessible but thorough
- Reference earlier blocks where helpful to build coherence and reinforce learning
- Use evidence-based cognitive strategies such as analogies, metaphors, chunking, and dual coding ONLY when helpful and appropriate
- Use clear, student-friendly language while maintaining accuracy
- Define all technical terms at first mention and reinforce understanding through examples

RESPONSE FORMAT (JSON only):
{
  "lesson": [
    {
      "type": "overview",
      "title": "Complete Overview of [Topic]",
      "content": "Comprehensive 400-600 word explanation covering the entire scope..."
    },
    {
      "type": "key_concepts", 
      "title": "Fundamental Concepts",
      "content": ["Concept 1: Full definition and explanation", "Concept 2: Full definition and explanation"]
    },
    {
      "type": "detailed_explanation",
      "title": "Subtopic 1 Deep Dive",
      "content": "200-300 words of comprehensive explanation..."
    }
  ]
}

IMPORTANT: 
- Output ONLY the JSON response - no additional text
- Ensure exactly 8-12 blocks total
- Each block must contain substantial and accurate educational content
- Cover the topic so thoroughly that a student would master it completely
"""

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

Create exactly 8-12 comprehensive teaching blocks that cover ALL aspects of this topic. Each block should be substantial and educational. Focus on complete mastery of the subject matter."""

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=0.2,  # Lower temperature for more consistent output
            max_tokens=4000,  # Increased for longer content
        )

        raw_output = response["choices"][0]["message"]["content"]
        
        # Clean up response to ensure it's valid JSON
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3]
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3]
        
        parsed_output = json.loads(raw_output)

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

            # Clean up content
            content = block.get("content", "")
            
            # Handle list content for key_concepts and similar types
            if block["type"] in ["key_concepts", "practice_questions", "memory_aids"]:
                if not isinstance(content, list):
                    # Convert string to list if needed
                    if isinstance(content, str):
                        content = [item.strip() for item in content.split('\n') if item.strip()]
                    else:
                        content = [str(content)]
                block["content"] = content
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

# --- Additional endpoint for quick concept review --- #
@router.post("/quick-review")
def generate_quick_review(req: TeachingRequest):
    """Generate condensed review content for last-minute studying"""
    try:
        quick_prompt = f"""
        Create exactly 6-8 condensed review blocks for: {req.description}
        
        Focus on:
        - Key facts and formulas
        - Most likely test questions
        - Critical concepts to remember
        - Quick memory aids
        
        Each block should be 200-300 words. Return in same JSON format.
        """

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": quick_prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=3000,
        )

        raw_output = response["choices"][0]["message"]["content"]
        
        # Clean up response
        raw_output = raw_output.strip()
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3]
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3]
            
        parsed_output = json.loads(raw_output)
        
        return {"lesson": parsed_output.get("lesson", [])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
