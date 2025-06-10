from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# === CORS Setup ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://arlo-study-craft.lovable.app",   # Production Lovable app
        "https://id-preview--405e367a-b787-41ce-904a-d1882e6a9b65.lovable.app",  # Lovable Preview (Editor mode)
        "http://localhost:10000",                  # Local dev (optional)
    ],
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
from backend.feynman_feedback import router as feynman_router 
from upload_pdf import router as upload_pdf_router
from blurting import router as blurting_router


# --- Include all routes ---
app.include_router(flashcard_router)
app.include_router(quiz_router)
app.include_router(study_session_router)
app.include_router(chatbot_router)
app.include_router(review_router)
app.include_router(feynman_router)
app.include_router(upload_pdf_router, prefix="/api")
app.include_router(blurting_router, prefix="/api")

# --- Root and health check ---
@app.get("/")
def root():
    return {"message": "ARLO backend is alive"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}

