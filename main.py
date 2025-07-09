from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware import Middleware
from dotenv import load_dotenv
import os
import jwt
import sys
import traceback

# Load environment variables from .env
load_dotenv()

# --- Middleware for extracting user info from JWT ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                decoded = jwt.decode(token, options={"verify_signature": False})
                request.state.user = decoded
            except Exception as e:
                print("❌ JWT decode error:", e)
                request.state.user = {}
        else:
            request.state.user = {}
        return await call_next(request)

# Create FastAPI app with middleware
app = FastAPI(middleware=[Middleware(AuthMiddleware)])

@app.middleware("http")
async def log_exceptions(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.write(f"🔥 UNCAUGHT EXCEPTION: {e}\n")
        raise

# === CORS Setup ===

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # ✅ Lovable actual preview URL
        "https://id-preview--c4e79f71-1738-4330-9bbd-c1a1b1fea023.lovable.app",

        # ✅ Lovable published domain
        "https://c4e79f71-1738-4330-9bbd-c1a1b1fea023.lovableproject.com",

        "https://arlo-ai-tutor.lovable.app",
        "https://lovable.app",
        "http://localhost:10000",
        "https://lovable.dev",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modular routers ---
from flashcard_generator import router as flashcard_router
from quiz import router as quiz_router
from study_session import router as study_session_router
from chatbot import router as chatbot_router
from review_sheet import router as review_router
from feynman_feedback import router as feynman_router 
from blurting import router as blurting_router
from context import router as context_router
from pdf_parser import router as pdf_parser_router
from teaching import router as teaching_router
from video_learning import router as video_learning_router

# --- Include all routes ---
app.include_router(flashcard_router, prefix="/api")
app.include_router(quiz_router, prefix="/api/quiz")
app.include_router(study_session_router, prefix="/api")
app.include_router(chatbot_router, prefix="/api")
app.include_router(review_router, prefix="/api")
app.include_router(feynman_router, prefix="/api")
app.include_router(blurting_router, prefix="/api")
app.include_router(context_router, prefix="/api")
app.include_router(pdf_parser_router, prefix="/api")
app.include_router(teaching_router, prefix="/api")
app.include_router(video_learning_router, prefix="/api")

# --- Root and health check ---
@app.get("/")
def root():
    return {"message": "ARLO backend is alive"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}
