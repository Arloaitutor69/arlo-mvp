import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_flashcards(topic: str, notes_text: str, difficulty: str = "medium", format: str = "Q&A"):
    prompt = f"""
You are a flashcard tutor generating study cards from the following topic and notes.

Topic: "{topic}"
Notes: "{notes_text}"

Difficulty: {difficulty}
Format: {format}

Use only one of these formats:
- "Q&A" → Basic question/answer
- "fill-in-the-blank" → Sentence with a missing term
- "multiple-choice" → (Not supported yet — just return Q&A for now)

Return output as a JSON array of objects:
[{{ "question": "...", "answer": "..." }}]

Only include flashcards, no explanation.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']
