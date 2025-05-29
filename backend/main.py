from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from session_planner import generate_study_plan

app = FastAPI()

# Define the structure of input data
class SessionRequest(BaseModel):
    subject: str
    duration_minutes: int
    notes_text: str = None

@app.get("/")
def read_root():
    return {"message": "Hello from ARLO backend!"}

@app.post("/generate-session")
def generate_session(request: SessionRequest):
    try:
        plan = generate_study_plan(
            subject=request.subject,
            duration_minutes=request.duration_minutes,
            notes_text=request.notes_text
        )
        return {"session_plan": plan}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
