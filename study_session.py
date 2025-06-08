from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import openai
import os

router = APIRouter()

# Load your OpenAI key from env
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== Request + Response Models ====
class StudySessionRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    difficulty: Optional[str] = "medium"
    target_level: Optional[str] = "high_school"

class SessionBlock(BaseModel):
    id: str
    phase: str
    start_min: int
    end_min: int
    description: str
    tool: Optional[str] = None
    payload: Optional[Dict] = None
    lovable_component: Optional[str] = None

class StudySessionResponse(BaseModel):
    session_id: str
    topic: str
    total_duration: int
    blocks: List[SessionBlock]

# ==== Helper: GPT Prompt Construction ====
def build_gpt_prompt(topic: str, duration: int, difficulty: str, level: str):
    return (
        f"""
        Create an optimal {duration}-minute study session on "{topic}" for a {level} student.
        Use techniques like: flashcards, quiz, Feynman explanation, blurting, active recall.
        Output JSON only: a list of blocks. Each block must include:
        - phase (string)
        - duration (int, in minutes)
        - description (string)
        - tool (string, e.g., 'flashcards', 'quiz', 'feynman', 'blurting', 'chatbot')
        - payload (dict, optional): config for tool (e.g., question_count)
        - lovable_component (string): which UI type to trigger (e.g., 'flashcard-carousel', 'quiz-form')
        Return only raw JSON.
        """
    )

# ==== GPT-Driven Study Plan Generator ====
@router.post("/api/study-session", response_model=StudySessionResponse)
def generate_study_session(data: StudySessionRequest):
    try:
        prompt = build_gpt_prompt(data.topic, data.duration, data.difficulty, data.target_level)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert study coach AI."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6
        )

        blocks_raw = completion.choices[0].message.content

        # Safely evaluate stringified JSON response
        import json
        try:
            blocks_json = json.loads(blocks_raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Failed to parse GPT response.")

        # Time tracker
        current = 0
        blocks = []

        for block in blocks_json:
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

        return StudySessionResponse(
            session_id=f"session_{uuid.uuid4().hex[:6]}",
            topic=data.topic,
            total_duration=current,
            blocks=blocks
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT error: {str(e)}")
