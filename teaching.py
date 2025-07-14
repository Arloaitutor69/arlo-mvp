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

# --- Output schema --- #
class TeachingBlock(BaseModel):
    type: Literal["overview", "key_concepts", "detailed_explanation", "examples", "test_strategies", "practice_questions", "memory_aids", "common_mistakes"]
    title: str
    content: Union[str, List[str]]
    importance: Optional[str] = None  # "High", "Medium", "Low" for test relevance

class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]
    estimated_study_time: str
    difficulty_level: str
    test_tips: List[str]

# --- Enhanced GPT Prompt --- #
GPT_SYSTEM_PROMPT = """
You are an elite test preparation tutor with expertise across all academic subjects. Your mission is to create comprehensive, test-focused learning content that maximizes student performance on exams.

CORE TEACHING PRINCIPLES:
1. TEST-CENTRIC APPROACH: Everything must be relevant to potential test questions or school curriculums 
2. COMPREHENSIVE COVERAGE: Cover all information relevant to that study block in its entirety. 
3. CLARITY & EFFICIENCY: Clear explanations without unnecessary fluff
4. Never include any informamtion that wouldn't be relevant to a class or test
5. Ensure that no relevant terms, examples, processes, edge cases, exceptions, or comparisons are omitted, even if they are considered more advanced or are typically covered in follow-up units.

CONTENT STRUCTURE REQUIREMENTS:
- Breakdown all information into digestible lessons that thourougly cover abolutely ALL info student might need to know for their class
- incorperate helpful examples, memory aids throughout (mnemonics, acronyms, visual associations but only when helpful relevant, otherwise avoid. 
- break down into 8 - 15 blocks which each explain a topic or set of related subtopics. By the end of all blocks, student should know all relevant info. 

CONTENT QUALITY STANDARDS:
- Each section should be substantial (300-450 words minimum)
- Teach the material the way an expert private tutor would, anticipating student confusion, using clear scaffolding, and connecting complex ideas to prior knowledge. 
- fully unpack and explain every subtopic, detail, and sub-skill that would be covered in a complete lesson or textbook unit on this topic. Do not summarize — explain each part separately in full instructional depth.
- Focus on understanding AND memorization where both are needed
- ensure teaching style is not overly intimidating and inaccesible, keep it clear and easy to understand

RESPONSE STRUCTURE:
{
  "lesson": [
    {
      "type": "overview",
      "title": "Test-Focused Overview",
      "content": "Strategic summary...",
      "importance": "High"
    },
    {
      "type": "key_concepts", 
      "title": "Essential Concepts",
      "content": ["Concept 1: Definition...", "Concept 2: Definition..."],
      "importance": "High"
    }
  ],
  "estimated_study_time": "2-3 hours",
  "difficulty_level": "Intermediate",
  "test_tips": ["Tip 1", "Tip 2", "Tip 3"]
}

Make content comprehensive, test-relevant, and immediately actionable for student success.
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
        
        user_prompt = f"{context_info}\nTopic to teach: {req.description}\n\nProvide comprehensive test-focused teaching content covering all essential aspects of this topic."

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,  # Lower temperature for more consistent, focused output
            max_tokens=3000,  # Increased for more comprehensive content
        )

        raw_output = response["choices"][0]["message"]["content"]
        parsed_output = json.loads(raw_output)

        # Enhanced validation and cleanup
        lesson_blocks = parsed_output.get("lesson", [])
        
        # Ensure comprehensive coverage by validating required block types
        required_types = ["overview", "key_concepts", "detailed_explanation", "examples"]
        existing_types = [block.get("type") for block in lesson_blocks]
        
        for block in lesson_blocks:
            if not isinstance(block, dict):
                continue

            # Validate and fix block structure
            if "type" not in block or block["type"] not in [
                "overview", "key_concepts", "detailed_explanation", "examples", 
                "test_strategies", "practice_questions", "memory_aids", "common_mistakes"
            ]:
                block["type"] = "detailed_explanation"

            # Ensure title exists
            if not block.get("title"):
                block["title"] = f"{block['type'].replace('_', ' ').title()}"

            # Clean up content structure
            if isinstance(block.get("content"), dict):
                # Handle nested content structures
                if "items" in block["content"]:
                    block["content"] = block["content"]["items"]
                elif "text" in block["content"]:
                    block["content"] = block["content"]["text"]
                else:
                    block["content"] = str(block["content"])

            # Ensure content is properly formatted
            content = block.get("content", "")
            
            # Handle list content - ensure all items are strings
            if isinstance(content, list):
                clean_content = []
                for item in content:
                    if isinstance(item, dict):
                        # Extract string content from dict
                        if "content" in item:
                            clean_content.append(str(item["content"]))
                        elif "text" in item:
                            clean_content.append(str(item["text"]))
                        else:
                            clean_content.append(str(item))
                    else:
                        clean_content.append(str(item))
                block["content"] = clean_content

            # Validate content type based on block type
            if block["type"] in ["key_concepts", "practice_questions", "memory_aids"]:
                # These should be lists
                if not isinstance(block.get("content"), list):
                    content_str = str(block.get("content", ""))
                    block["content"] = [item.strip() for item in content_str.split('\n') if item.strip()]
            else:
                # These should be strings
                if isinstance(block.get("content"), list):
                    # Convert list to string, ensuring all items are strings
                    string_items = [str(item) for item in block["content"]]
                    block["content"] = "\n\n".join(string_items)

            # Set importance if not provided
            if "importance" not in block:
                high_importance_types = ["overview", "key_concepts", "test_strategies"]
                block["importance"] = "High" if block["type"] in high_importance_types else "Medium"

        # Ensure required fields exist in response
        if "estimated_study_time" not in parsed_output:
            parsed_output["estimated_study_time"] = "2-3 hours"
        
        if "difficulty_level" not in parsed_output:
            parsed_output["difficulty_level"] = "Intermediate"
        
        if "test_tips" not in parsed_output:
            parsed_output["test_tips"] = [
                "Review key concepts multiple times",
                "Practice with sample questions",
                "Focus on understanding, not just memorization"
            ]

        return parsed_output

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
        Create a condensed, high-yield review of: {req.description}
        
        Focus on:
        - Key facts and formulas
        - Most likely test questions
        - Critical concepts to remember
        - Quick memory aids
        
        Return in same JSON format but more concise.
        """

        messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "user", "content": quick_prompt}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )

        raw_output = response["choices"][0]["message"]["content"]
        parsed_output = json.loads(raw_output)
        
        # Add quick review indicator
        parsed_output["estimated_study_time"] = "30-45 minutes"
        parsed_output["format"] = "Quick Review"
        
        return parsed_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
