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
# Schemas
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
# JSON Schemas for structured outputs
# ---------------------------
CHATBOT_SCHEMA = {
    "name": "chatbot_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "message": {
                "type": "string",
                "minLength": 10
            },
            "follow_up_question": {
                "type": "string",
                "minLength": 10
            },
            "learning_concepts": {
                "type": "array",
                "maxItems": 5,
                "items": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 30
                }
            },
            "confidence_level": {
                "type": "string",
                "enum": ["low", "medium", "high"]
            }
        },
        "required": ["message", "follow_up_question", "learning_concepts", "confidence_level"],
        "additionalProperties": False
    }
}

HELP_SCHEMA = {
    "name": "help_response",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "explanation": {
                "type": "string",
                "minLength": 50
            }
        },
        "required": ["explanation"],
        "additionalProperties": False
    }
}

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
    """Build an enhanced prompt for tutoring with better context and examples"""
    
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

Provide a clear, informative response that directly answers the student's question. Always include a thoughtful follow-up question to continue learning. Identify key learning concepts covered in your response. Be concise but thorough.

EXAMPLE QUALITY RESPONSES:

For Biology question about photosynthesis:
Message: "Photosynthesis is how plants convert sunlight into energy through two main stages: light reactions (in thylakoids) and the Calvin cycle (in stroma). The overall equation is 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. Light reactions capture energy and produce ATP/NADPH, while Calvin cycle uses these to convert CO2 into glucose. Chlorophyll absorbs light energy, mainly red and blue wavelengths, which is why plants appear green."
Follow-up: "Would you like to explore how different wavelengths of light affect photosynthesis efficiency?"
Learning concepts: ["photosynthesis", "light reactions", "Calvin cycle", "chlorophyll", "ATP/NADPH"]

For Math question about quadratic equations:
Message: "Quadratic equations have the form ax² + bx + c = 0. You can solve them using the quadratic formula: x = (-b ± √(b² - 4ac)) / 2a. The discriminant (b² - 4ac) tells you about solutions: positive = 2 real roots, zero = 1 repeated root, negative = 2 complex roots. For example, x² - 5x + 6 = 0 has a = 1, b = -5, c = 6, giving x = (5 ± 1) / 2, so x = 3 or x = 2."
Follow-up: "Want to try factoring this equation instead of using the formula?"
Learning concepts: ["quadratic formula", "discriminant", "real roots", "factoring"]

Your response should help the student understand the topic better and encourage further exploration."""

    return prompt

def build_help_prompt(content: str) -> str:
    """Build simplified help prompt that explains content in clear paragraphs with examples"""
    
    prompt = f"""You are Arlo, an AI tutor. Explain the following content in a clear, easy-to-understand way using 1-2 concise paragraphs.

CONTENT TO EXPLAIN:
{content}

Instructions:
- Write in paragraph form, not numbered lists or bullet points
- Use simple, clear language that's easier to understand than the original
- Focus on the main concepts and their significance
- Keep it concise but informative
- Include examples only if they help clarify the concept

EXAMPLE QUALITY EXPLANATIONS:

For complex scientific text about DNA replication:
"DNA replication is the process where cells copy their genetic material before dividing. Think of it like making a perfect photocopy of an instruction manual. The double helix unwinds, and special enzymes called DNA polymerases read each strand and build complementary new strands, following base-pairing rules (A with T, G with C). This ensures each new cell gets an exact copy of the genetic instructions."

For mathematical concept about derivatives:
"Derivatives measure how fast something is changing at any given moment. Imagine you're driving and look at your speedometer - that shows your derivative of distance with respect to time. In math, if you have a curve on a graph, the derivative at any point tells you the slope of the tangent line there, which represents the rate of change at that exact moment."

Provide a clear explanation that helps the student understand this content better."""

    return prompt

def call_gpt_sync(prompt: str, schema: dict, fallback_response: dict) -> dict:
    """Synchronous GPT call using GPT-5-nano structured outputs"""
    try:
        messages = [
            {"role": "system", "content": "You are Arlo, an AI tutor. Be concise, clear, and helpful. Always provide educational content with concrete examples and specific details."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            reasoning_effort="low"
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return fallback_response

async def call_gpt_async(prompt: str, schema: dict, fallback_response: dict) -> dict:
    """Async wrapper for GPT call"""
    return call_gpt_sync(prompt, schema, fallback_response)

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
    logger.info("Enhanced chatbot request received")
    try:
        user_id = extract_user_id(request, data)
        prompt = build_enhanced_prompt(data)
        
        # Fallback response for error cases
        fallback_chatbot_response = {
            "message": "I'm having trouble right now. Please try rephrasing your question.",
            "follow_up_question": "What would you like to learn more about?",
            "learning_concepts": [],
            "confidence_level": "low"
        }
        
        # Async GPT call for better performance
        gpt_reply = await call_gpt_async(prompt, CHATBOT_SCHEMA, fallback_chatbot_response)
        
        # Extract data from structured response
        message = gpt_reply.get("message", "I'm having trouble right now.")
        follow_up = gpt_reply.get("follow_up_question", f"What would you like to explore further about {data.topic}?")
        concepts_covered = gpt_reply.get("learning_concepts", [])
        confidence_level = gpt_reply.get("confidence_level", "medium")
        
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
            message=message,
            follow_up_question=follow_up,
            context_update_required=context_update_needed,
            learning_concepts_covered=concepts_covered,
            confidence_level=confidence_level
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
    logger.info("Help request received")
    try:
        prompt = build_help_prompt(data.content)
        
        # Fallback response for error cases
        fallback_help_response = {
            "explanation": "I'm having trouble providing help right now. Please try rephrasing your question or breaking it into smaller parts."
        }
        
        # Use GPT with structured output
        response = await call_gpt_async(prompt, HELP_SCHEMA, fallback_help_response)
        
        return HelpResponse(explanation=response.get("explanation", fallback_help_response["explanation"]))
        
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
