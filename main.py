from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup — allow anything for testing (we'll tighten it later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins just for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def health_check():
    return {"status": "ok"}
