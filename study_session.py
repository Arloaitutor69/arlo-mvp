import openai
import os
import json
import uuid
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

router = APIRouter()

# --- Models ---
class StudyPlanRequest(BaseModel):
    topic: str
    duration: int  # in minutes
    details: Optional[str] = None  # user-submitted focus or goals
    difficulty: Optional[str] = "medium"
    target_level: Optional[str] = "high_school"
    parsed_text: Optional[str] = None  # optional notes/textbook input

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
def build_gpt_prompt(topic: str, details: Optional[str], duration: int, level: str, parsed_text: Optional[str]) -> str:
    detail_text = f"\nThe student mentioned specific goals:\n\"{details.strip()}\"" if details else ""
    source_text = f"\n\nUse the following source material as the primary base:\n{parsed_text[:3000]}..." if parsed_text else ""

    return (
        "You are ARLO, an AI-powered tutor designing a structured study session.\n\n"
        f"The student has {duration} minutes to study the subject: \"{topic}\"."
        f"{detail_text}{source_text}\n\n"
        "Instructions:\n"
        "- First, break the topic into 3–6 instructional units (as in a mini curriculum)\n"
        "- Assign 1 learning technique per unit\n"
        "- Choose from: flashcards, quiz, feynman, blurting, arlo_teaching\n"
        "- Begin with the most important units if time is tight\n"
        "- End with a review or self-reflection block if possible\n"
        "- Suggest the best Pomodoro format (e.g. 25/5, 50/10)\n"
        "- Use only clear, parsable JSON format.\n\n"
        "Return ONLY this JSON:\n"
        "{{\n"
        "  \"units_to_cover\": [\"...\"],\n"
        "  \"pomodoro\": \"25/5\",\n"
        "  \"techniques\": [\"...\"],\n"
        "  \"blocks\": [\n"
        "    {{\n"
        "      \"unit\": \"...\",\n"
        "      \"technique\": \"flashcards\",\n"
        "      \"description\": \"...\",\n"
        "      \"duration\": 8\n"
        "    }}\n"
        "  ]\n"
        "}}"
    )

# --- Endpoint ---
@router.post("/api/plan", response_model=StudyPlanResponse)
def generate_plan(data: StudyPlanRequest):
    try:
        prompt = build_gpt_prompt(data.topic, data.details, data.duration, data.target_level, data.parsed_text)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a world-class curriculum planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        raw = completion.choices[0].message.content.strip()
        if raw.startswith("```"):  # strip markdown fences if GPT adds them
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

            # context update per block with curriculum and planning info
            try:
                requests.post(f"{CONTEXT_API}/api/context/update", json={
                    "source": "user:54a623b9-9804-456b-9ae5-4fc9e048859d",
                    "current_topic": f"{unit} — {desc}",
                    "learning_event": {
                        "concept": unit,
                        "phase": tech,
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Planned block: {desc}",
                        "repetition_count": 0,
                        "review_scheduled": False
                    }
                })
            except Exception as e:
                print(f"⚠️ Context update failed for {unit}: {e}")

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
