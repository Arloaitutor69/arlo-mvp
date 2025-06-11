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



@app.middleware("http")
async def log_exceptions(request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.write(f"🔥 UNCAUGHT EXCEPTION: {e}\n")
        raise

# Create FastAPI app with middleware
app = FastAPI(middleware=[Middleware(AuthMiddleware)])

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

# --- Modular routers ---
from flashcard_generator import router as flashcard_router
from quiz import router as quiz_router
from study_session import router as study_session_router
from chatbot import router as chatbot_router
from review_sheet import router as review_router
from backend.feynman_feedback import router as feynman_router 
from upload_pdf import router as upload_pdf_router
from blurting import router as blurting_router
from context import router as context_router

# --- Include all routes ---
app.include_router(flashcard_router)
app.include_router(quiz_router)
app.include_router(study_session_router)
app.include_router(chatbot_router)
app.include_router(review_router)
app.include_router(feynman_router)
app.include_router(upload_pdf_router, prefix="/api")
app.include_router(blurting_router, prefix="/api")
app.include_router(context_router, prefix="/api")

# --- Root and health check ---
@app.get("/")
def root():
    return {"message": "ARLO backend is alive"}

@app.get("/ping")
def health_check():
    return {"status": "ok"}
