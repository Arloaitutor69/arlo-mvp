from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import fitz  # PyMuPDF
import openai
import os

router = APIRouter()

# Load your OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

@router.post("/pdf/parse", response_class=PlainTextResponse)
async def parse_pdf(file: UploadFile = File(...)):
    # Step 1: Check file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Step 2: Read and parse up to 4 pages from PDF
    try:
        contents = await file.read()
        doc = fitz.open(stream=contents, filetype="pdf")

        text = ""
        for page_num in range(min(4, len(doc))):
            page = doc[page_num]
            page_text = page.get_text("text")
            text += page_text + "\n"

        if not text.strip():
            raise ValueError("PDF contains no readable text")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    # Step 3: Send to OpenAI for summarization
    try:
        prompt = (
            "You are an AI tutor helping design a personalized study plan from a student's document.\n\n"
            "Your task is to extract only the *academic content* from the following text. Ignore all non-learning info "
            "(such as professors, textbooks, grading, or scheduling).\n\n"
            "Summarize the key learning material in a way that supports building a detailed study curriculum. "
            "Include:\n"
            "- Core topics and subtopics\n"
            "- Important definitions or terms\n"
            f"{text.strip()}"
        )


        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You summarize academic content clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=512
        )

        summary = response["choices"][0]["message"]["content"].strip()
        return summary

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT summarization failed: {str(e)}")
