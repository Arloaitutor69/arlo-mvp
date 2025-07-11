# ENHANCED TUTORING CHATBOT MODULE

from fastapi import APIRouter, FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
import openai
import os
import logging
import requests
import asyncio
import aiohttp
from datetime import datetime
import json
import re

# ---------------------------
# Setup
# ---------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
CONTEXT_API = os.getenv("CONTEXT_API_BASE", "https://arlo-mvp-2.onrender.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")

app = FastAPI()
router = APIRouter()

# ---------------------------
# Simplified Schemas
# ---------------------------
class ChatbotInput(BaseModel):
    user_input: str
    topic: str
    target_level: Optional[str] = "medium"
    message_history: Optional[List[Dict[str, str]]] = []
    user_id: Optional[str] = None

class HelpInput(BaseModel):
    content: str

class ChatbotResponse(BaseModel):
    message: str
    follow_up_question: Optional[str] = None

class HelpResponse(BaseModel):
    explanation: str
    key_concepts: List[str]

# ---------------------------
# Enhanced Helpers
# ---------------------------
def extract_user_id(request: Request, data: ChatbotInput) -> str:
    if request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    elif data.source and str(data.source).startswith("user:"):
        return data.source.replace("user:", "")
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

def build_simple_prompt(data: ChatbotInput) -> str:
    """Build a lean, fast prompt for tutoring"""
    
    # Get last 3 user and GPT responses
    recent_history = []
    for msg in data.message_history[-5:]:  # Last 6 messages (3 user + 3 GPT)
        recent_history.append(f"{msg['role']}: {msg['content']}")
    
    history = "\n".join(recent_history) if recent_history else "No previous conversation."
    
    prompt = f"""You are Arlo, an AI tutor. Be concise, informative, and helpful.

Topic: {data.topic}
Level: {data.target_level}

Recent conversation:
{history}

Student: "{data.user_input}"

Provide a clear, informative response that directly answers the student's question. Include one follow-up question to continue learning. Be concise but thorough."""

    return prompt

def build_help_prompt(content: str) -> str:
    """Build simple help prompt that analyzes content type"""
    
    prompt = f"""You are Arlo, an AI tutor. Analyze the content below and explain it step by step.

CONTENT TO EXPLAIN:
{content}

Instructions:
- Provide a clear, step-by-step explanation appropriate for the content type
- For quiz questions: explain the correct approach and key concepts
- For flashcards: break down the concept thoroughly
- For complex topics: use simple analogies and examples
- Be concise but comprehensive
- Focus on understanding, not just memorization

Your explanation:"""

    return prompt

async def call_gpt_async(prompt: str, max_tokens: int = 250) -> str:
    """Fast, lean GPT call"""
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are Arlo, an AI tutor. Be concise, clear, and helpful."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            request_timeout=10
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return "I'm having trouble right now. Please try rephrasing your question."

def extract_key_concepts(response: str) -> List[str]:
    """Quick concept extraction"""
    concepts = []
    # Look for key terms in quotes or after colons
    quoted = re.findall(r'"([^"]*)"', response)
    concepts.extend([c for c in quoted if 3 < len(c) < 30])
    
    # Look for definition patterns
    definitions = re.findall(r'(\w+(?:\s+\w+){0,2})\s+(?:is|are|means)', response, re.IGNORECASE)
    concepts.extend([d.strip() for d in definitions if 3 < len(d.strip()) < 30])
    
    return list(set(concepts))[:3]  # Return top 3

# ---------------------------
# Main Chatbot Route (Enhanced)
# ---------------------------
@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_handler(request: Request, data: ChatbotInput, background_tasks: BackgroundTasks):
    logger.info("Enhanced chatbot request received")
    try:
        user_id = extract_user_id(request, data)
        prompt = build_enhanced_prompt(data)
        
        # Async GPT call for better performance
        gpt_reply = await call_gpt_async(prompt, max_tokens=450)
        
        # Extract learning concepts
        concepts_covered = extract_learning_concepts(gpt_reply)
        
        # Generate natural follow-up question
        follow_up = None
        if "?" not in gpt_reply:  # Add follow-up if response doesn't already contain a question
            follow_up = f"What would you like to explore further about {data.session_summary.topic}?"
        
        # Determine if context update is needed
        context_update_needed = len(concepts_covered) > 0 or any(
            keyword in data.user_input.lower() 
            for keyword in ["understand", "learn", "explain", "what", "how", "why"]
        )
        
        # Schedule async context update
        if context_update_needed:
            background_tasks.add_task(
                update_context_async, 
                user_id, 
                {
                    "concepts_covered": concepts_covered,
                    "last_interaction": datetime.now().isoformat(),
                    "topic": data.session_summary.topic
                }
            )
        
        return ChatbotResponse(
            message=gpt_reply,
            follow_up_question=follow_up,
            context_update_required=context_update_needed,
            learning_concepts_covered=concepts_covered,
            confidence_level="high" if len(gpt_reply) > 100 else "medium"
        )
        
    except Exception as e:
        logger.error(f"Enhanced chatbot handler failed: {e}")
        raise HTTPException(status_code=500, detail="I'm having trouble right now. Please try again.")

# ---------------------------
# New Help Router
# ---------------------------
@router.post("/chatbot/help", response_model=HelpResponse)
async def help_handler(data: HelpInput):
    """Handle specific help requests for quiz questions, flashcards, feynman or blurting study technique."""
    logger.info(f"Help request received for {data.question_type}")
    try:
        prompt = build_help_prompt(data)
        
        # Use higher token limit for detailed explanations
        response = await call_gpt_async(prompt, max_tokens=500)
        
        # Extract key concepts
        key_concepts = extract_learning_concepts(response)
        
        # Generate step-by-step breakdown if it's a process-oriented question
        steps = []
        if data.question_type in ["quiz", "feynman"] and any(word in response.lower() for word in ["first", "then", "next", "finally", "step"]):
            # Simple step extraction
            step_matches = re.findall(r'(?:Step \d+|First|Then|Next|Finally)[:\-]?\s*([^.!?]*[.!?])', response)
            steps = [step.strip() for step in step_matches if step.strip()]
        
        # Generate related topics
        related_topics = []
        if data.topic:
            # This would ideally use a knowledge graph or curriculum mapping
            topic_lower = data.topic.lower()
            if "math" in topic_lower:
                related_topics = ["algebra", "geometry", "calculus", "statistics"]
            elif "science" in topic_lower:
                related_topics = ["physics", "chemistry", "biology", "scientific method"]
            elif "history" in topic_lower:
                related_topics = ["historical analysis", "primary sources", "timelines", "cause and effect"]
            
            # Filter to relevant topics
            related_topics = [t for t in related_topics if t != data.topic.lower()][:3]
        
        # Practice suggestions based on question type
        practice_suggestions = {
            "quiz": ["Review similar questions", "Create your own practice questions", "Explain the concept to someone else"],
            "flashcard": ["Use spaced repetition", "Create related flashcards", "Practice active recall"],
            "feynman": ["Try explaining without looking", "Use analogies", "Test your understanding with examples"],
            "blurting": ["Practice timed recall", "Organize information hierarchically", "Create concept maps"]
        }
        
        return HelpResponse(
            explanation=response,
            key_concepts=key_concepts,
            step_by_step=steps if steps else None,
            related_topics=related_topics if related_topics else None,
            practice_suggestions=practice_suggestions.get(data.question_type, [])
        )
        
    except Exception as e:
        logger.error(f"Help handler failed: {e}")
        raise HTTPException(status_code=500, detail="I'm having trouble providing help right now. Please try again.")

# ---------------------------
# Context Save Endpoint (Enhanced)
# ---------------------------
@router.post("/chatbot/save")
async def save_chat_context(payload: Dict[str, Any]):
    """Enhanced context saving with better error handling"""
    try:
        logger.info("Saving enhanced chatbot context")
        
        # Add timestamp
        payload["timestamp"] = datetime.now().isoformat()
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_API}/api/context/update",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return {"status": "success", "message": "Context saved successfully"}
                else:
                    logger.warning(f"Context save returned status {response.status}")
                    return {"status": "warning", "message": f"Context save returned status {response.status}"}
                    
    except Exception as e:
        logger.error(f"Context save failed: {e}")
        return {"status": "error", "detail": str(e)}

# ---------------------------
# Health Check Endpoint
# ---------------------------
@router.get("/chatbot/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test OpenAI API
        await call_gpt_async("Test", max_tokens=10)
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# ---------------------------
# Include in App
# ---------------------------
app.include_router(router)
