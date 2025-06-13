import openai
import os
import json
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

# --- Config ---
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
    review_sheet: List[str]
    optional_priming_video: Optional[str]


# --- Helper Functions ---
def build_gpt_prompt(topic: str, details: Optional[str], duration: int, level: str) -> str:
    detail_text = f"\nThe student mentioned specific focus areas or goals:\n\"{details.strip()}\"." if details else ""
    return f"""
You are ARLO, an AI-powered study coach using research-backed learning science.

The student has {duration} minutes to study the subject:\n\"{topic}\".{detail_text}

---

**Step 1: Break down the subject into the necessary units, concepts, or modules a student must understand to master this subject. Use your knowledge of curriculum guides, Khan Academy structures, textbooks, and prerequisite scaffolding. Return this list in the field `units_to_cover`. This should be specific and complete. Do NOT skip any required ideas.**

**Step 2: Create a detailed study plan that ensures ALL units are covered.** Use only the most relevant techniques from this list:

- Active Recall (quiz questions, blurting)
- Spaced Repetition / Leitner Flashcards
- Mind Mapping
- Feynman Technique (teach-back)
- Interleaved Practice (only if multiple topics are mentioned)
- Worked Examples / Socratic problem-solving
- Visual Sketching or Concept Mapping
- Pomodoro Time Management (e.g., 25/5 or 50/10)
- Daily Review Summary
- Short YouTube Primers (for abstract/difficult concepts)

Choose techniques based on content type:

- **Conceptual**: Feynman, mind maps, YouTube primers, active recall
- **Procedural**: worked examples, Socratic steps, visual intuition
- **Memorization-heavy**: flashcards, spaced repetition, active recall
- **Multiple topics**: interleaved practice

---

Respond ONLY with a valid JSON object, no markdown, no explanations.
Start your reply with `{{` and end with `}}`
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
            temperature=0.6
        )

        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```json"):
            raw = raw.split("```", 1)[-1].strip()

        parsed = json.loads(raw)
        session_id = f"session_{uuid.uuid4().hex[:6]}"

        # Convert tasks into study blocks
        blocks = []
        total_time = 0

        for idx, task in enumerate(parsed.get("tasks", [])):
            block_id = f"block_{uuid.uuid4().hex[:6]}"
            unit = parsed["units_to_cover"][min(idx, len(parsed["units_to_cover"]) - 1)]
            technique = parsed["techniques"][min(idx, len(parsed["techniques"]) - 1)]

            blocks.append(StudyBlock(
                id=block_id,
                unit=unit,
                technique=technique,
                phase=technique.lower().replace(" ", "_"),
                tool=technique.lower().replace(" ", "_"),
                lovable_component="text-block",  # can be mapped more specifically later
                duration=8,  # default for now
                description=task,
                position=idx,
                custom=False,
                user_notes=None
            ))
            total_time += 8

        return StudyPlanResponse(
            session_id=session_id,
            topic=data.topic,
            total_duration=total_time,
            pomodoro=parsed.get("pomodoro", "25/5"),
            units_to_cover=parsed["units_to_cover"],
            techniques=parsed["techniques"],
            blocks=blocks,
            review_sheet=parsed.get("review_sheet", []),
            optional_priming_video=parsed.get("optional_priming_video")
        )

    except Exception as e:
        print("🔥 Study plan generation failed:", e)
        raise HTTPException(status_code=500, detail="Failed to generate study plan")
