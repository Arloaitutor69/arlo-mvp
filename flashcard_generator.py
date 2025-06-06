import openai
import os

# Load OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_flashcards(topic: str, notes_text: str, difficulty: str = "medium", format: str = "Q&A"):
    # Strict prompt to ensure clean JSON output from GPT
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

Return ONLY a valid JSON array of objects in this format:
[
  {{ "question": "What is ...?", "answer": "..." }},
  {{ "question": "...?", "answer": "..." }}
]

Do not include explanations, headers, or any other text — just return the JSON array.
"""

    # Use the updated OpenAI v1.0+ syntax
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    # Return the plain text from the assistant's reply
    return response.choices[0].message.content
