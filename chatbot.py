# ENHANCED TUTORING CHATBOT MODULE

from fastapi import APIRouter, FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from openai import OpenAI
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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
    user_id: Optional[str] = None

class ChatbotResponse(BaseModel):
    message: str
    follow_up_question: Optional[str] = None
    context_update_required: Optional[bool] = False
    learning_concepts_covered: Optional[List[str]] = []
    confidence_level: Optional[str] = "medium"

class HelpResponse(BaseModel):
    explanation: str

# ---------------------------
# Enhanced Helpers
# ---------------------------
def extract_user_id(request: Request, data: ChatbotInput) -> str:
    if request.headers.get("x-user-id"):
        return request.headers["x-user-id"]
    elif data.user_id:
        return data.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user_id in request")

def build_enhanced_prompt(data: ChatbotInput) -> str:
    """Build an enhanced prompt for tutoring with better context"""
    
    # Get last 5 messages for context
    recent_history = []
    for msg in data.message_history[-5:]:
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
    """Build simplified help prompt that explains content in clear paragraphs"""
    
    prompt = f"""You are Arlo, an AI tutor. Explain the following content in a clear, easy-to-understand way using 1-2 concise paragraphs.

CONTENT TO EXPLAIN:
{content}

Instructions:
- Write in paragraph form, not numbered lists or bullet points
- Use simple, clear language that's easier to understand than the original
- Focus on the main concepts and their significance
- Keep it concise but informative
- Include examples only if they help clarify the concept

Your explanation:"""

    return prompt

def clean_response(raw_response: str) -> str:
    """Clean and validate GPT response"""
    if not raw_response:
        return "I'm having trouble right now. Please try rephrasing your question."
    
    # Remove any potential markdown formatting
    cleaned = raw_response.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    
    return cleaned

async def call_gpt_async(prompt: str, max_response_length: str = "concise") -> str:
    """Fast, lean GPT-5 Nano call with enhanced error handling"""
    try:
        # GPT-5 Nano optimized call
        response = client.chat.completions.create(
            model="gpt-5-nano",  # GPT-5 Nano model
            messages=[
                {"role": "system", "content": "You are Arlo, an AI tutor. Be concise, clear, and helpful. Provide direct answers without unnecessary formatting."},
                {"role": "user", "content": prompt}
            ],
            reasoning_effort="low",  # Speed optimization for GPT-5 Nano
            top_p=0.9
        )
        
        raw_content = response.choices[0].message.content
        cleaned_response = clean_response(raw_content)
        
        logger.info(f"GPT-5 Nano response length: {len(cleaned_response)} chars")
        return cleaned_response
        
    except Exception as e:
        logger.error(f"GPT-5 Nano call failed: {e}")
        return "I'm having trouble right now. Please try rephrasing your question."

def extract_learning_concepts(response: str) -> List[str]:
    """Extract key learning concepts from response"""
    concepts = []
    # Look for key terms in quotes or after colons
    quoted = re.findall(r'"([^"]*)"', response)
    concepts.extend([c for c in quoted if 3 < len(c) < 30])
    
    # Look for definition patterns
    definitions = re.findall(r'(\w+(?:\s+\w+){0,2})\s+(?:is|are|means)', response, re.IGNORECASE)
    concepts.extend([d.strip() for d in definitions if 3 < len(d.strip()) < 30])
    
    return list(set(concepts))[:3]  # Return top 3

async def update_context_async(user_id: str, context_data: Dict[str, Any]):
    """Update user context asynchronously"""
    try:
        # Fixed payload structure with required 'source' field
        payload = {
            "user_id": user_id,
            "source": "chatbot",  # Added required source field
            "concepts_covered": context_data.get("concepts_covered", []),
            "last_interaction": context_data.get("last_interaction", datetime.now().isoformat()),
            "topic": context_data.get("topic", ""),
            "session_type": "chatbot",
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_API}/api/context/update",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    logger.info(f"Context updated for user {user_id}")
                else:
                    response_text = await response.text()
                    logger.warning(f"Context update failed with status {response.status}: {response_text}")
    except Exception as e:
        logger.error(f"Context update failed: {e}")

# ---------------------------
# Main Chatbot Route (Enhanced)
# ---------------------------
@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot_handler(request: Request, data: ChatbotInput, background_tasks: BackgroundTasks):
    logger.info("Enhanced chatbot request received - using GPT-5 Nano")
    try:
        user_id = extract_user_id(request, data)
        prompt = build_enhanced_prompt(data)
        
        # Async GPT-5 Nano call for better performance
        gpt_reply = await call_gpt_async(prompt, max_response_length="detailed")
        
        # Extract learning concepts
        concepts_covered = extract_learning_concepts(gpt_reply)
        
        # Generate natural follow-up question
        follow_up = None
        if "?" not in gpt_reply:  # Add follow-up if response doesn't already contain a question
            follow_up = f"What would you like to explore further about {data.topic}?"
        
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
                    "topic": data.topic
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
# Simplified Help Router
# ---------------------------
@router.post("/chatbot/help", response_model=HelpResponse)
async def help_handler(request: Request, data: HelpInput):
    """Simplified help endpoint - just explains the content"""
    logger.info("Help request received - using GPT-5 Nano")
    try:
        prompt = build_help_prompt(data.content)
        
        # Use GPT-5 Nano for detailed explanations
        response = await call_gpt_async(prompt, max_response_length="detailed")
        
        return HelpResponse(explanation=response)
        
    except Exception as e:
        logger.error(f"Help handler failed: {e}")
        raise HTTPException(status_code=500, detail="I'm having trouble providing help right now. Please try again.")

# ---------------------------
# Context Save Endpoint (Enhanced)
# ---------------------------
@router.post("/chatbot/save")
async def save_chat_context(request: Request, payload: Dict[str, Any]):
    """Enhanced context saving with better error handling"""
    try:
        logger.info("Saving enhanced chatbot context")
        
        # Extract user_id from request or payload
        user_id = request.headers.get("x-user-id") or payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="Missing user_id")
        
        # Structure payload to match API expectations with required 'source' field
        formatted_payload = {
            "user_id": user_id,
            "source": "chatbot",  # Added required source field
            "concepts_covered": payload.get("concepts_covered", []),
            "last_interaction": payload.get("last_interaction", datetime.now().isoformat()),
            "topic": payload.get("topic", ""),
            "session_type": payload.get("session_type", "chatbot"),
            "message_history": payload.get("message_history", []),
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CONTEXT_API}/api/context/update",
                json=formatted_payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return {"status": "success", "message": "Context saved successfully"}
                else:
                    response_text = await response.text()
                    logger.warning(f"Context save returned status {response.status}: {response_text}")
                    return {"status": "warning", "message": f"Context save returned status {response.status}", "details": response_text}
                    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context save failed: {e}")
        return {"status": "error", "detail": str(e)}


# ---------------------------
# Include in App
# ---------------------------
app.include_router(router)
