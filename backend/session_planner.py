import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_study_plan(subject: str, duration_minutes: int, notes_text: str = None):
    prompt = f"""
You are ARLO, an AI study planner that uses cognitive science.

A student has {duration_minutes} minutes to study {subject}.
Their notes say: {notes_text or "no notes provided"}.

You must consider these proven study techniques (select only the most relevant for this session):

- Active Recall
- Spaced Repetition / Leitner Flashcards
- Mind Mapping
- Feynman Technique (teach-back)
- Blurting
- Interleaved Practice
- Chunking
- Mnemonics
- Memory Palace
- Elaborative Interrogation ("why" questions)
- Pomodoro Time Blocking
- Daily Review Sheets

Now, generate a session plan using the best techniques for this topic and time window.

Return this in JSON format:
{{
  "pomodoro": "Best interval (e.g. 25/5)",
  "techniques": ["Selected techniques from the list above"],
  "tasks": ["Step-by-step activities using the techniques"],
  "review_sheet": ["Brief summary for bedtime review"]
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response['choices'][0]['message']['content']
    return content

