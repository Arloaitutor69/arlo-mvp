import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_feynman_feedback(topic: str, user_explanation: str):
    prompt = f"""
You are an AI tutor helping a student understand a topic by evaluating their explanation.

Topic: "{topic}"
Student's Explanation: "{user_explanation}"

Your job:
- Give 1–2 sentences of friendly feedback.
- Be supportive and encouraging, but point out any unclear or inaccurate parts.
- Ask 2 follow-up questions that guide the student to reflect or go deeper.

Return only valid JSON in this format:
{{
  "feedback": "your feedback here",
  "follow_up_questions": ["question 1", "question 2"]
}}
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt }],
        temperature=0.7
    )

    return response['choices'][0]['message']['content']
