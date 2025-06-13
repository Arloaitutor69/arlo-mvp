import openai
import os
import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# --- Models ---
class StudyPlanRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    details: Optional[str] = None
    difficulty: Optional[str] = "medium"
    target_level: Optional[str] = "high_school"

class StudyBlock(BaseModel):
    id: str
    unit: str
    technique: str
    phase: str
    tool: str
    lovable_component: str
    duration: int
    description: str
    position: int
    custom: bool = False
    user_notes: Optional[str] = None
    payload: Optional[Dict] = None

class StudyPlanResponse(BaseModel):
    session_id: str
    topic: str
    total_duration: int
    pomodoro: str
    units_to_cover: List[str]
    techniques: List[str]
    blocks: List[StudyBlock]

# --- Helper: Optimized GPT Prompt ---
def build_gpt_prompt(topic: str, details: Optional[str], duration: int, level: str) -> str:
    detail_text = f"\nThe student also mentioned these specific focus areas or goals:\n\"{details.strip()}\"." if details else ""
    return f"""
You are ARLO, an AI study coach. The student has {duration} minutes to study the topic: \"{topic}\".{detail_text}

Break the subject into necessary units using textbook-level precision. Return them under `units_to_cover`.

Then return a detailed JSON study plan assigning ONE technique to EACH unit (from this list only):
- arlo_teaching
- flashcards
- quiz
- feynman
- blurting

Match each unit to ONE technique only and describe the learning activity in one sentence. Allocate roughly equal time per unit.

Set `pomodoro` to the best fitting time format (e.g., 25/5).

Respond with a **complete valid JSON object only** like this:

[[
{{
  "units_to_cover": ["..."],
  "pomodoro": "25/5",
  "techniques": ["..."],
  "tasks": ["..."],
  "total_duration": 45
}}
]]

NO extra text, markdown, or comments. Fill ALL fields with your best guess if unsure.
"""

# --- Endpoint ---
@router.post("/api/plan", response_model=StudyPlanResponse)
def generate_plan(data: StudyPlanRequest):
    try:
        prompt = build_gpt_prompt(data.topic, data.details, data.duration, data.target_level)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a world-class curriculum planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 1)[-1].strip()

        parsed = json.loads(raw)

        session_id = f"session_{uuid.uuid4().hex[:6]}"
        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        tasks = parsed.get("tasks", [])
        pomodoro = parsed.get("pomodoro", "25/5")
        total_duration = parsed.get("total_duration", data.duration)

        blocks = []
        minutes_per_block = max(1, total_duration // max(1, len(tasks)))

        for idx, task in enumerate(tasks):
            block_id = f"block_{uuid.uuid4().hex[:6]}"
            technique = techniques[min(idx, len(techniques)-1)]
            unit = units[min(idx, len(units)-1)] if units else "General"

            blocks.append(StudyBlock(
                id=block_id,
                unit=unit,
                technique=technique,
                phase=technique,
                tool=technique,
                lovable_component="text-block",
                duration=minutes_per_block,
                description=task,
                position=idx
            ))

        return StudyPlanResponse(
            session_id=session_id,
            topic=data.topic,
            total_duration=total_duration,
            pomodoro=pomodoro,
            units_to_cover=units,
            techniques=techniques,
            blocks=blocks
        )

    except Exception as e:
        print("🔥 Study plan generation failed:", e)
        raise HTTPException(status_code=500, detail="Failed to generate study plan")
