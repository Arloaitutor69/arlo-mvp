import openai
import os
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_study_plan(subject: str, duration_minutes: int, notes_text: str = None):
    prompt = f"""
You are ARLO, an AI-powered study coach using research-backed learning science.

A student has {duration_minutes} minutes to study the subject:
"{subject}"

Their notes say: "{notes_text or 'no notes provided'}"

---

**Step 1: Break down the subject into the necessary units, concepts, or modules a student must understand to master this subject. Use your knowledge of curriculum guides, Khan Academy structures, textbooks, and prerequisite scaffolding. Return this list in the field `units_to_cover`. This should be specific and complete. Do NOT skip any required ideas.**

---

**Step 2: Create a detailed study plan that ensures ALL units are covered.** Use only the most relevant techniques from this list:

- Active Recall (quiz questions, blurting)
- Spaced Repetition / Leitner Flashcards
- Mind Mapping
- Feynman Technique (teach-back)
- Interleaved Practice (only if multiple topics are mentioned)
- Worked Examples / Socratic problem-solving
- Visual Sketching or Concept Mapping
- Pomodoro Time Management (e.g., 25/5)
- Daily Review Summary
- Short YouTube Primers (for abstract/difficult concepts)

Choose techniques based on the type of content:

- **Conceptual**: Feynman, mind maps, YouTube primers, active recall
- **Procedural**: worked examples, Socratic steps, visual intuition
- **Memorization-heavy**: flashcards, spaced repetition, active recall
- **Multiple topics**: interleaved practice

---

Output the following JSON **only**:

{{
  "units_to_cover": ["Unit 1", "Unit 2", "..."],
  "pomodoro": "Best interval format (e.g. 25/5 or 50/10)",
  "techniques": ["List of most relevant techniques used"],
  "tasks": ["Step-by-step learning tasks using those techniques, covering ALL units"],
  "review_sheet": ["3–5 bedtime review bullet points based on the above content"],
  "optional_priming_video": "YouTube search phrase for concept explainer, or null if not needed"
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']
