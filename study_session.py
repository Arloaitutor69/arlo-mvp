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
    source_text = f"\n\nUse the following source material as your primary reference:\n{parsed_text[:3000]}..." if parsed_text else ""

    return (
        "You are ARLO, an AI-powered tutor creating a deeply structured study session.\n\n"
        f"The student has {duration} minutes to study the topic: \"{topic}\" at a **{level} level**."
        f"{detail_text}{source_text}\n\n"
        "Instructions:\n"
        "- Break the topic into 4 to 6 specific instructional units (like a mini curriculum).\n"
        "- Assign **exactly one learning technique** to each block: flashcards, quiz, feynman, blurting, or arlo_teaching.\n"
        "- For each block, generate:\n"
        "  • A unit title\n"
        "  • The assigned technique\n"
        "  • A detailed and content-rich `description` of what is to be studied, explained, or practiced\n"
        "    - Include key subtopics, core terms, processes, or examples to cover\n"
        "    - Include concrete comprehension goals (e.g., 'Be able to label and explain the function of each organelle')\n"
        "    - Avoid generic phrasing like 'You will now review...' or 'Take a quiz on biology'. Instead, describe the actual content\n"
        "  • A reasonable duration (e.g., 8–15 min) that fits the complexity of the unit\n"
        "- The `description` field is the only source of context the AI tutor and frontend will receive, so it must be specific and self-contained.\n"
        "- Use clear language, even for complex content. If possible, tailor difficulty to the specified academic level.\n"
        "- Format response as strict JSON ONLY — no markdown, headers, or extra commentary.\n\n"
        "Example format:\n"
        "{\n"
        "  \"units_to_cover\": [\"Structure of the Cell Membrane\", \"Photosynthesis Pathway\"],\n"
        "  \"pomodoro\": \"25/5\",\n"
        "  \"techniques\": [\"flashcards\", \"quiz\"],\n"
        "  \"blocks\": [\n"
        "    {\n"
        "      \"unit\": \"Structure of the Cell Membrane\",\n"
        "      \"technique\": \"flashcards\",\n"
        "      \"description\": \"Review phospholipid bilayer components (hydrophilic heads, hydrophobic tails), protein channels, cholesterol's role in fluidity, and embedded glycoproteins. Comprehension goal: Be able to diagram and label all parts with their function.\",\n"
        "      \"duration\": 10\n"
        "    }\n"
        "  ]\n"
        "}"
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
        any_block_logged = False

        for idx, item in enumerate(blocks_json):
            unit = item.get("unit", "General")
            tech = item.get("technique", "feynman")
            desc = item.get("description", "Study the topic")
            mins = item.get("duration", 8)
            block_id = f"block_{uuid.uuid4().hex[:6]}"

            try:
                payload = {
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
                }
                print("📤 Sending to context:\n", json.dumps(payload, indent=2))
                resp = requests.post(f"{CONTEXT_API}/api/context/update", json=payload)
                print("✅ Context update response:", resp.status_code, resp.text)
                if resp.status_code == 200:
                    any_block_logged = True
            except Exception as e:
                print(f"⚠️ Context update failed for {unit}:\n{e}")

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

        # Final synthesis trigger bundled with a meaningful event
        if any_block_logged:
            try:
                print("🧠 Final plan context + synthesis trigger...")
                final_payload = {
                    "source": "user:54a623b9-9804-456b-9ae5-4fc9e048859d",
                    "current_topic": f"Full study plan: {data.topic}",
                    "learning_event": {
                        "concept": data.topic,
                        "phase": "planning",
                        "confidence": None,
                        "depth": None,
                        "source_summary": f"Structured plan generated with {len(blocks)} study blocks.",
                        "repetition_count": 0,
                        "review_scheduled": False
                    },
                    "trigger_synthesis": True
                }
                synth = requests.post(f"{CONTEXT_API}/api/context/update", json=final_payload)
                print("🧠 Synthesis response:", synth.status_code, synth.text)
            except Exception as e:
                print("⚠️ Failed to trigger synthesis:", e)

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
