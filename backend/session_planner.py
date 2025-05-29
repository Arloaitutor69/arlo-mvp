import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_study_plan(subject: str, duration_minutes: int, notes_text: str = None):
    prompt = f"""
You are ARLO, an AI-powered study coach using proven cognitive science strategies.

A student has {duration_minutes} minutes to study the following subject:
"{subject}"

Their notes say: "{notes_text or 'no notes provided'}"

---

🎯 Your task is to generate a study plan using only the most relevant methods from this list:

- Active Recall (quiz questions, blurting)
- Spaced Repetition / Leitner Flashcards
- Mind Mapping
- Feynman Technique (teach-back)
- Interleaved Practice (only if multiple topics are mentioned)
- Worked Examples / Socratic problem-solving
- Visual Sketching or Concept Mapping
- Pomodoro Time Management (e.g., 25/5)
- Daily Review Summary
- Short YouTube Primers (if topic is conceptually difficult or abstract)

---

📚 Choose techniques based on content type:

- For **conceptual topics** (e.g. biology, philosophy, history):
  → use Feynman, mind maps, YouTube primers, active recall.

- For **procedural/mathematical topics** (e.g. dynamical systems, calculus, physics):
  → use worked examples, Socratic Q&A, visual intuition, step-by-step problem-solving.
  → avoid flashcards or mind maps unless highly customized.

- For **memorization-heavy topics** (e.g. vocab, anatomy, dates):
  → use flashcards, spaced repetition, active recall.

- For **multiple topics** in one session:
  → use interleaved practice to mix problem types or concepts.

---

Return this JSON structure exactly (do NOT explain anything outside this structure):

{{
  "pomodoro": "Best interval format (e.g. 25/5 or 50/10)",
  "techniques": ["List of most relevant techniques used"],
  "tasks": ["Step-by-step learning tasks using those techniques"],
  "review_sheet": ["3–5 bedtime review bullet points"],
  "optional_priming_video": "Short YouTube search phrase to find a concept explainer (or null if not needed)"
}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

