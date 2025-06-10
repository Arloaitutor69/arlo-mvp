# upload_pdf.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import io

router = APIRouter()

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, extract its text content, and return the raw text.
    Intended to be used before generating a study session.
    """
    # Ensure correct file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read file into memory
        contents = await file.read()
        reader = PdfReader(io.BytesIO(contents))

        # Extract text from each page
        extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)

        if not extracted_text.strip():
            raise ValueError("No readable text found in PDF.")

        # Return the extracted text to the frontend
        return JSONResponse(content={"extracted_text": extracted_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
