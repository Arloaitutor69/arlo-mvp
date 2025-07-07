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
            "You are an academic summarizer. Summarize the following PDF content into a concise, clear overview "
            "that captures the main ideas and topics. Don't miss any key details, facts, or definitions.\n\n"
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
