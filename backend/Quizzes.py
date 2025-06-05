import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_chat_response(message: str) -> str:
    prompt = f"You are an AI tutor. Respond helpfully to: {message}"
    result = openai.ChatCompletion.create(
        model="gpt-3.5",
        messages=[{"role": "user", "content": prompt}]
    )
    return result["choices"][0]["message"]["content"]

def generate_quiz(topic: str, notes_text: str, num_questions: int = 5) -> list:
    prompt = f"""
Create a {num_questions}-question multiple-choice quiz on the topic '{topic}' using the notes below:

{notes_text}

Return a JSON list where each item has:
- 'question': the quiz question
- 'choices': a list of 3-4 answer options
- 'answer': the correct choice from the list
"""
    result = openai.ChatCompletion.create(
        model="gpt-3.5",
        messages=[{"role": "user", "content": prompt}]
    )
    return result["choices"][0]["message"]["content"]
