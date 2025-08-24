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
    type: Literal["overview", "key_concepts", "detailed_explanation", "examples", "test_strategies", "practice_questions", "memory_aids", "common_mistakes"]
    title: str
    content: str

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- JSON Schema for structured outputs --- #
TEACHING_SCHEMA = {
    "name": "teaching_response",
    "schema": {
        "type": "object",
        "properties": {
            "lesson": {
                "type": "array",
                "minItems": 10,
                "maxItems": 10,
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
                                "test_strategies", 
                                "practice_questions", 
                                "memory_aids", 
                                "common_mistakes"
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

# --- Enhanced GPT System Prompt --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor creating comprehensive, engaging, easy-to-understand learning content. Create exactly 10 teaching blocks that thoroughly cover ALL aspects of the requested topic.

CRITICAL REQUIREMENTS:
1. Create EXACTLY 10 blocks
2. Each block must have substantial content (minimum 50 characters)
3. Use proper block types from the allowed list
4. Focus ONLY on information relevant to a class/test, not tangential information

TEACHING BLOCK STRUCTURE:
- Block 1: "overview" - what student will learn, main questions, learning goals
- Blocks 2-8: Mix of "detailed_explanation", "key_concepts", "examples" covering core content
- Block 9: "test_strategies", "common_mistakes", or "practice_questions"
- Block 10: "memory_aids" - summary and key takeaways

CONTENT QUALITY STANDARDS:
- Explain concepts in extremely easy-to-understand, casual language
- Use clear scaffolding and connect to prior knowledge
- Use analogies when helpful
- Define all technical terms at first mention
- Student-friendly language while maintaining accuracy
- Vary teaching strategies and structure between blocks

BLOCK TYPES TO USE:
- "overview": Introduction, learning goals, main questions
- "key_concepts": Essential terms and principles
- "detailed_explanation": In-depth explanations of concepts
- "examples": Real-world applications and illustrations
- "test_strategies": Study tips and exam approaches
- "practice_questions": Sample problems or quiz questions
- "memory_aids": Mnemonics, summaries, key takeaways
- "common_mistakes": Typical errors students make

Remember: Create engaging, educational content that builds understanding step by step."""

@router.post("/teaching", response_model=TeachingResponse)
def generate_teaching_content(req: TeachingRequest):
    try:
        # Build context information
        context_parts = []
        if req.subject:
            context_parts.append(f"Subject: {req.subject}")
        if req.level:
            context_parts.append(f"Academic Level: {req.level}")
        if req.test_type:
            context_parts.append(f"Test/Exam Type: {req.test_type}")
        
        context_info = "\n".join(context_parts) if context_parts else "General academic context"
        
        # Create user prompt
        user_prompt = f"""{context_info}

Topic to teach: {req.description}

Create a comprehensive 10-block lesson that covers all important aspects of this topic. Make it engaging, educational, and appropriate for the specified context."""

        # Prepare messages
        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Make API call with structured outputs
        response = client.chat.completions.create(
            model="gpt-5-nano",
            temperature=0,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": TEACHING_SCHEMA,
                "strict": True
            },
            reasoning_effort="low"
        )

        # Parse the guaranteed valid JSON response
        raw_content = response.choices[0].message.content
        parsed_data = json.loads(raw_content)
        
        # Convert to Pydantic models for additional validation
        lesson_blocks = [
            TeachingBlock(
                type=block["type"],
                title=block["title"],
                content=block["content"]
            )
            for block in parsed_data["lesson"]
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
