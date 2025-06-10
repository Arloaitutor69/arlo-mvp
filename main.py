from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# --- CORS config for Lovable frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://405e367a-b787-41ce-904a-d1882e6a9b65.lovableproject.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# --- Modular routers (FastAPI style) ---
from flashcard_generator import router as flashcard_router
from quiz import router as quiz_router
from study_session import router as study_session_router
from chatbot import router as chatbot_router
from review_sheet import router as review_router
from feynman_feedback import router as feynman_router  

# --- Include all routes ---
app.include_router(flashcard_router)
app.include_router(quiz_router)
app.include_router(study_session_router)
app.include_router(chatbot_router)
app.include_router(review_router)
app.include_router(feynman_router)

# --- Root and health check ---
@app.get("/")
def root():
    return {"message": "ARLO backend is alive"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}
