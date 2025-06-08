from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import openai
import os
import json

# Load environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# Request format
class StudySessionRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    difficulty: Optional[str] = "medium"
    target_level: Optional[str] = "high_school"

# Each study block
class SessionBlock(BaseModel):
    id: str
    phase: str
    start_min: int
    end_min: int
    description: str
    tool: Optional[str] = None
    payload: Optional[Dict] = None
    lovable_component: Optional[str] = None

# Final response
class StudySessionResponse(BaseModel):
    session_id: str
    topic: str
    total_duration: int
    blocks: List[SessionBlock]

# Helper: Build GPT system message
def build_gpt_prompt(topic: str, duration: int, difficulty: str, level: str) -> str:
    return f"""
Create a {duration}-minute study session on the topic \"{topic}\" for a {level} student.

Use evidence-based techniques like:
- flashcards
- quiz
- Feynman technique
- blurting
- active recall

Output a JSON array. Each item should be:
- phase: one of [\"priming\", \"flashcards\", \"quiz\", \"feynman\", \"blurting\", \"review\"]
- duration: in minutes
- description: what the student will do
- tool: matching the backend module (e.g., \"flashcards\", \"quiz\", \"feynman\", \"blurting\", \"chatbot\")
- payload: optional dictionary (e.g., question_count, target_level)
- lovable_component: UI component name (e.g., \"flashcard-carousel\", \"quiz-form\", \"chat-bubble\")

Return raw JSON only, no explanation.
"""

@router.post("/api/study-session", response_model=StudySessionResponse)
def generate_study_session(data: StudySessionRequest):
    print("📩 Received study session request:")
    print(data)

    try:
        prompt = build_gpt_prompt(data.topic, data.duration, data.difficulty, data.target_level)
        print("📤 Prompt sent to GPT:\n", prompt)

        # OpenAI call
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a world-class AI tutor and study coach."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )
        gpt_raw = completion.choices[0].message.content

        # 🛠 Remove Markdown-style code block wrapper if present
        if gpt_raw.strip().startswith("```json") or gpt_raw.strip().startswith("```"):
            gpt_raw = gpt_raw.strip().split("\n", 1)[1].rsplit("```", 1)[0].strip()
        
        print("🤖 GPT raw response:\n", gpt_raw)
        
        # Parse GPT output
        try:
            block_list = json.loads(gpt_raw)
        except json.JSONDecodeError as e:
            print("❌ JSON parse error:", str(e))
            raise HTTPException(status_code=500, detail="Invalid JSON from GPT")
        
        # Time tracking
        current = 0
        blocks = []

        for block in block_list:
            duration = block.get("duration", 5)
            blocks.append(SessionBlock(
                id=f"block_{uuid.uuid4().hex[:6]}",
                phase=block.get("phase"),
                start_min=current,
                end_min=current + duration,
                description=block.get("description"),
                tool=block.get("tool"),
                payload=block.get("payload"),
                lovable_component=block.get("lovable_component")
            ))
            current += duration

        print("✅ Session plan built with", len(blocks), "blocks.")

        return StudySessionResponse(
            session_id=f"session_{uuid.uuid4().hex[:6]}",
            topic=data.topic,
            total_duration=current,
            blocks=blocks
        )

    except Exception as e:
        print("🔥 Unexpected error:", str(e))
        raise HTTPException(status_code=500, detail="Study session generation failed.")
