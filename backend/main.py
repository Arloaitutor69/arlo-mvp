# backend/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
import openai

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set. Please export it in your terminal.")

@app.get("/")
def root():
    return {"status": "ARLO backend is running"}

@app.post("/chat")
async def chat(request: Request):
    print("[INFO] /chat endpoint hit")
    data = await request.json()
    user_input = data.get("message", "")
    print(f"[DEBUG] user_input = {user_input}")

    if not user_input:
        return {"response": "Empty message received."}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor."},
                {"role": "user", "content": user_input}
            ]
        )

        reply = response["choices"][0]["message"]["content"]
        print("[DEBUG] reply:", reply)

        return {"response": reply}

    except Exception as e:
        print("[ERROR] OpenAI failure:", e)
        return {"response": f"Error occurred: {str(e)}"}
