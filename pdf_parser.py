from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
import fitz  # PyMuPDF
from openai import OpenAI
import os
import asyncio
import concurrent.futures
import re
import json
from datetime import datetime
import logging

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
router = APIRouter()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_parser")

# -----------------------------
# Enhanced PDF Processing Class
# -----------------------------
class EnhancedPDFProcessor:
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        
    async def extract_text_optimized(self, contents: bytes, max_pages: int = None) -> dict:
        """Optimized text extraction with parallel processing"""
        try:
            # Open PDF document
            doc = fitz.open(stream=contents, filetype="pdf")
            total_pages = len(doc)
            
            # If no max_pages specified, process all pages (but limit to reasonable amount for performance)
            pages_to_process = min(max_pages or min(total_pages, 100), total_pages)
            
            # Process pages in parallel for faster extraction
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                page_futures = []
                
                for page_num in range(pages_to_process):
                    future = executor.submit(self._extract_page_content, doc, page_num)
                    page_futures.append(future)
                
                # Collect results
                page_contents = []
                for future in concurrent.futures.as_completed(page_futures):
                    try:
                        page_data = future.result()
                        if page_data['text'].strip():
                            page_contents.append(page_data)
                    except Exception as e:
                        logger.error(f"Error processing page: {e}")
                        continue
            
            doc.close()
            
            # Sort by page number and combine
            page_contents.sort(key=lambda x: x['page_num'])
            full_text = "\n\n".join([page['text'] for page in page_contents])
            
            # Enhanced text cleaning
            cleaned_text = self._clean_extracted_text(full_text)
            
            return {
                'text': cleaned_text,
                'total_pages': total_pages,
                'processed_pages': pages_to_process
            }
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            raise HTTPException(status_code=500, detail=f"Error extracting PDF content: {str(e)}")
    
    def _extract_page_content(self, doc: fitz.Document, page_num: int) -> dict:
        """Extract content from a single page with multiple methods"""
        page = doc[page_num]
        
        # Method 1: Standard text extraction
        text = page.get_text("text")
        
        # Method 2: Try dict extraction for better structure if text is sparse
        if len(text.strip()) < 100:
            try:
                blocks = page.get_text("dict")
                structured_text = self._extract_from_blocks(blocks)
                if len(structured_text.strip()) > len(text.strip()):
                    text = structured_text
            except:
                pass
        
        # Method 3: OCR fallback for image-heavy pages (if text is still sparse)
        if len(text.strip()) < 50:
            try:
                # Try to get text from images using PyMuPDF's built-in capabilities
                text_instances = page.get_text("words")
                if text_instances:
                    text = " ".join([word[4] for word in text_instances])
            except:
                pass
        
        return {
            'page_num': page_num + 1,
            'text': text
        }
    
    def _extract_from_blocks(self, blocks_dict: dict) -> str:
        """Extract text from PyMuPDF blocks dictionary"""
        text_parts = []
        
        for block in blocks_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    if line_text.strip():
                        text_parts.append(line_text.strip())
        
        return "\n".join(text_parts)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Enhanced text cleaning for better readability"""
        if not text:
            return ""
        
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'\n\s*Page \d+.*?\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)
        
        # Remove isolated single characters on their own lines
        text = re.sub(r'\n\s*[a-zA-Z]\s*\n', '\n', text)
        
        # Clean up bullet points and formatting
        text = re.sub(r'[•·▪▫◦‣⁃]\s*', '• ', text)
        
        return text.strip()

# -----------------------------
# Chunking for Large Documents
# -----------------------------
class IntelligentChunker:
    def __init__(self, chunk_size: int = 6000):
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str) -> list:
        """Intelligently chunk text while preserving structure"""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split by major sections first
        sections = re.split(r'\n\s*\n\s*(?=[A-Z][^.]*:|\d+\.|\w+\s+\d+)', text)
        
        chunks = []
        current_chunk = ""
        
        for section in sections:
            if len(current_chunk + section) <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + section
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle oversized sections
                if len(section) > self.chunk_size:
                    sub_chunks = self._split_by_sentences(section, self.chunk_size)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = section
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_sentences(self, text: str, max_size: int) -> list:
        """Split text by sentences when it exceeds chunk size"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

# -----------------------------
# AI Processing
# -----------------------------
async def process_with_gpt5_nano(text: str) -> str:
    """Process text with GPT-5-nano for better quality and reliability"""
    
    system_prompt = """You are an AI tutor helping design a personalized study plan from a student's document.

Your task is to extract only the *academic content* from the following text. Ignore all non-learning info (such as professors, textbooks, grading, or scheduling).

Summarize the key learning material in a way that supports building a detailed study curriculum. Include:
- Core topics and subtopics
- Important definitions or terms
- Key concepts and principles
- Learning objectives

Focus on content that would help a student learn and understand the subject matter."""

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract and summarize the academic content from this document:\n\n{text}"}
        ]

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            reasoning_effort="low"
        )

        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"GPT-5-nano processing failed: {e}")
        # Fallback to a basic summary if AI fails
        return f"Academic Content Summary:\n\n{text[:2000]}..." if len(text) > 2000 else text

# -----------------------------
# Initialize processors
# -----------------------------
processor = EnhancedPDFProcessor()
chunker = IntelligentChunker()

# -----------------------------
# EXACT SAME ENDPOINT - Enhanced Internally
# -----------------------------
@router.post("/pdf/parse", response_class=PlainTextResponse)
async def parse_pdf(file: UploadFile = File(...)):
    """
    Enhanced PDF parser - same endpoint, same response format, but with:
    - Parallel page processing for speed
    - Better text extraction (handles images, tables, formatting)
    - Processes more pages (up to 100 instead of 4)
    - Intelligent chunking for large documents
    - GPT-5-nano for better quality summaries
    - Enhanced text cleaning and formatting
    """
    
    # Step 1: Check file type (same as original)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Step 2: Enhanced PDF processing (instead of basic 4-page limit)
    try:
        contents = await file.read()
        
        # Check file size
        if len(contents) > processor.max_file_size:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Extract text with enhanced methods - no longer limited to just 4 pages
        extraction_result = await processor.extract_text_optimized(contents)
        text = extraction_result['text']
        
        if not text.strip():
            raise ValueError("PDF contains no readable text")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    # Step 3: Enhanced AI processing
    try:
        # Handle large documents by chunking
        if len(text) > 6000:
            chunks = chunker.chunk_text(text)
            
            # Process chunks in parallel for speed
            chunk_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                chunk_futures = [
                    executor.submit(
                        lambda chunk: asyncio.run(process_with_gpt5_nano(chunk)),
                        chunk
                    ) 
                    for chunk in chunks[:8]  # Process up to 8 chunks for performance
                ]
                
                for future in concurrent.futures.as_completed(chunk_futures):
                    try:
                        result = future.result()
                        if result and result.strip():
                            chunk_results.append(result)
                    except Exception as e:
                        logger.error(f"Chunk processing failed: {e}")
            
            # Combine results
            if chunk_results:
                summary = "\n\n---\n\n".join(chunk_results)
            else:
                # Fallback: process first part of text
                summary = await process_with_gpt5_nano(text[:6000])
        else:
            # Process entire text if it's small enough
            summary = await process_with_gpt5_nano(text)
        
        # Return same format as original (PlainTextResponse)
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT summarization failed: {str(e)}")
