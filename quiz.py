from fastapi import APIRouter

router = APIRouter()

@router.get("/api/quiz/test")
def quiz_health_check():
    return {"status": "quiz router loads"}
