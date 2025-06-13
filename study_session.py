import openai
import os
import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

# --- Models ---
class StudyPlanRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    details: Optional[str] = None  # user-submitted focus or goals
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

# --- Helper Functions ---
def build_gpt_prompt(topic: str, details: Optional[str], duration: int, level: str) -> str:
    detail_text = f"\nThe student mentioned specific focus areas or goals:\n\"{details.strip()}\"." if details else ""
    return f"""
You are ARLO, an AI-powered study coach using research-backed learning science.

The student has {duration} minutes to study the subject:\n\"{topic}\".{detail_text}

Use only these techniques:
- arlo_teaching
- flashcards
- quiz
- feynman
- blurting

---

Output the following JSON object:
{{
  "units_to_cover": ["Unit 1", "Unit 2", "..."],
  "pomodoro": "Best Pomodoro interval like 25/5",
  "techniques": ["flashcards", "quiz", "feynman"],
  "blocks": [
    {{
      "unit": "Name of the unit",
      "technique": "one of: flashcards, quiz, feynman, blurting, arlo_teaching",
      "description": "What the student will do",
      "duration": 8
    }}
  ]
}}

Respond with ONLY valid JSON (no markdown or explanations). Make your best guess for all fields. Start with {{ and end with }}.
"""

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
        print("🤖 GPT RAW:", raw)

        if raw.startswith("```"):
            raw = raw.split("```", 1)[-1].strip()

        parsed = json.loads(raw)
        session_id = f"session_{uuid.uuid4().hex[:6]}"

        units = parsed.get("units_to_cover", [])
        techniques = parsed.get("techniques", [])
        blocks_json = parsed.get("blocks", [])
        pomodoro = parsed.get("pomodoro", "25/5")

        blocks = []
        total_time = 0

        for idx, item in enumerate(blocks_json):
            unit = item.get("unit", "General")
            tech = item.get("technique", "feynman")
            desc = item.get("description", "Study the topic")
            mins = item.get("duration", 8)
            block_id = f"block_{uuid.uuid4().hex[:6]}"

            blocks.append(StudyBlock(
                id=block_id,
                unit=unit,
                technique=tech,
                phase=tech,
                tool=tech,
                lovable_component="text-block",
                duration=mins,
                description=desc,
                position=idx
            ))
            total_time += mins

        return StudyPlanResponse(
            session_id=session_id,
            topic=data.topic,
            total_duration=total_time,
            pomodoro=pomodoro,
            units_to_cover=units,
            techniques=techniques,
            blocks=blocks
        )

    except Exception as e:
        print("🔥 Study plan generation failed:", e)
        raise HTTPException(status_code=500, detail="Failed to generate study plan")
