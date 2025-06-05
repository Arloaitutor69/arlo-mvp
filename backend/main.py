from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env (for local dev)
load_dotenv()

# App instance
app = FastAPI()

# Allow cross-origin requests from Lovable frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with Lovable domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example endpoint
@app.get("/")
def root():
    return {"message": "Backend is live on Render!"}

# Example Supabase call
@app.get("/supabase-users")
def get_users():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    response = requests.get(
        f"{supabase_url}/rest/v1/users",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        },
    )
    return response.json()

