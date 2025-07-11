from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timedelta, timezone
import json
import openai
import os
from supabase import create_client, Client
import requests
from collections import defaultdict
import threading
import asyncio
import hashlib
from dataclasses import dataclass
import math

# ---------------------------
# Configuration
# ---------------------------
CONFIDENCE_DECAY_RATE = 0.95  # Daily decay multiplier
SYNTHESIS_THRESHOLD = 5  # Trigger synthesis every N learning events
MAX_CONTEXT_HISTORY = 50  # Concepts to keep in memory
CACHE_TTL_MINUTES = 5
STALE_THRESHOLD_MINUTES = 2

# ---------------------------
# In-memory Context Cache
# ---------------------------
context_cache: Dict[str, Dict[str, Any]] = {}
synthesis_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

@dataclass
class ConceptMemory:
    """Efficient concept representation for memory"""
    concept: str
    confidence: float
    depth: str
    last_seen: datetime
    repetition_count: int
    sources: List[str]
    next_review: Optional[datetime] = None
    
    def calculate_retention(self) -> float:
        """Calculate retention based on time decay"""
        days_since = (datetime.now() - self.last_seen).days
        return self.confidence * (CONFIDENCE_DECAY_RATE ** days_since)
    
    def schedule_review(self) -> None:
        """Schedule next review using spaced repetition"""
        interval_days = min(2 ** self.repetition_count, 30)  # Cap at 30 days
        self.next_review = datetime.now() + timedelta(days=interval_days)

# ------------------------------
# Supabase Client
# ------------------------------
supabase: Optional[Client] = None

def get_supabase() -> Client:
    global supabase
    if not supabase:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE not set")
        supabase = create_client(url, key)
    return supabase

# ------------------------------
# Router and Models
# ------------------------------
router = APIRouter()

class LearningEvent(BaseModel):
    concept: str
    phase: str
    confidence: Optional[float] = 0.5
    depth: Optional[Literal['shallow', 'intermediate', 'deep']] = 'shallow'
    source_summary: Optional[str] = None
    repetition_count: Optional[int] = 1
    review_scheduled: Optional[bool] = False

class ContextUpdate(BaseModel):
    user_id: Optional[str] = None
    current_topic: Optional[str] = None
    user_goals: Optional[List[str]] = None
    preferred_learning_styles: Optional[List[str]] = None
    weak_areas: Optional[List[str]] = None
    emphasized_facts: Optional[List[str]] = None
    review_queue: Optional[List[str]] = None
    learning_event: Optional[LearningEvent] = None
    source: str
    feedback_flag: Optional[bool] = False
    trigger_synthesis: Optional[bool] = False

class ContextResetRequest(BaseModel):
    user_id: str

# ------------------------------
# Core Memory Functions
# ------------------------------

def aggregate_learning_events(events: List[Dict]) -> Dict[str, ConceptMemory]:
    """Efficiently aggregate learning events by concept"""
    concept_map: Dict[str, ConceptMemory] = {}
    
    for event in events:
        le = event.get("learning_event", {})
        if not le.get("concept"):
            continue
            
        concept = le["concept"]
        confidence = le.get("confidence", 0.5)
        depth = le.get("depth", "shallow")
        source = event.get("source", "unknown")
        timestamp = datetime.fromisoformat(event.get("timestamp", datetime.now().isoformat()))
        
        if concept in concept_map:
            # Update existing concept
            memory = concept_map[concept]
            memory.confidence = max(memory.confidence, confidence)  # Take best confidence
            memory.depth = max(memory.depth, depth, key=lambda x: ["shallow", "intermediate", "deep"].index(x))
            memory.last_seen = max(memory.last_seen, timestamp)
            memory.repetition_count += 1
            if source not in memory.sources:
                memory.sources.append(source)
        else:
            # Create new concept memory
            concept_map[concept] = ConceptMemory(
                concept=concept,
                confidence=confidence,
                depth=depth,
                last_seen=timestamp,
                repetition_count=1,
                sources=[source]
            )
    
    return concept_map

def identify_weak_areas(concepts: Dict[str, ConceptMemory], threshold: float = 0.6) -> List[str]:
    """Algorithmically identify weak areas based on retention"""
    weak_concepts = []
    
    for concept, memory in concepts.items():
        current_retention = memory.calculate_retention()
        if current_retention < threshold:
            weak_concepts.append(concept)
    
    return sorted(weak_concepts, key=lambda c: concepts[c].calculate_retention())

def generate_review_queue(concepts: Dict[str, ConceptMemory]) -> List[str]:
    """Generate review queue based on spaced repetition"""
    now = datetime.now()
    due_for_review = []
    
    for concept, memory in concepts.items():
        if memory.next_review is None:
            memory.schedule_review()
        
        if memory.next_review <= now:
            due_for_review.append(concept)
    
    # Sort by urgency (oldest due first)
    return sorted(due_for_review, key=lambda c: concepts[c].next_review or now)

def create_context_hash(events: List[Dict]) -> str:
    """Create hash to detect if synthesis is needed"""
    content = json.dumps(events, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()

def should_trigger_synthesis(user_id: str, events: List[Dict]) -> bool:
    """Smart synthesis triggering"""
    # Check if enough new events
    if len(events) < SYNTHESIS_THRESHOLD:
        return False
    
    # Check if content has changed significantly
    current_hash = create_context_hash(events)
    cache_key = f"synthesis_hash_{user_id}"
    
    if cache_key in context_cache:
        last_hash = context_cache[cache_key].get("hash")
        if current_hash == last_hash:
            return False
    
    context_cache[cache_key] = {"hash": current_hash}
    return True

def synthesize_context_efficient(user_id: str, recent_events: List[Dict]) -> Dict[str, Any]:
    """Efficient context synthesis without GPT for most cases"""
    try:
        # Aggregate learning events efficiently
        concepts = aggregate_learning_events(recent_events)
        
        # Extract metadata from most recent events
        meta_fields = {
            "current_topic": None,
            "user_goals": [],
            "preferred_learning_styles": [],
            "emphasized_facts": []
        }
        
        for event in reversed(recent_events):  # Most recent first
            for field in meta_fields:
                if event.get(field) and not meta_fields[field]:
                    meta_fields[field] = event[field]
        
        # Generate derived insights
        weak_areas = identify_weak_areas(concepts)
        review_queue = generate_review_queue(concepts)
        
        # Convert concepts to serializable format
        learning_history = []
        for concept, memory in concepts.items():
            learning_history.append({
                "concept": memory.concept,
                "confidence": memory.calculate_retention(),
                "depth": memory.depth,
                "repetition_count": memory.repetition_count,
                "sources": memory.sources[:3],  # Limit to 3 sources
                "last_seen": memory.last_seen.isoformat(),
                "next_review": memory.next_review.isoformat() if memory.next_review else None
            })
        
        # Sort by relevance (recent + high confidence)
        learning_history.sort(key=lambda x: (
            datetime.fromisoformat(x["last_seen"]),
            x["confidence"]
        ), reverse=True)
        
        return {
            **meta_fields,
            "weak_areas": weak_areas[:10],  # Limit to top 10
            "review_queue": review_queue[:10],  # Limit to top 10
            "learning_history": learning_history[:MAX_CONTEXT_HISTORY]
        }
        
    except Exception as e:
        print(f"❌ Efficient synthesis failed: {e}")
        return fallback_to_gpt_synthesis(user_id, recent_events)

def fallback_to_gpt_synthesis(user_id: str, events: List[Dict]) -> Dict[str, Any]:
    """GPT synthesis as fallback for complex cases"""
    try:
        # Create minimal prompt focusing on key insights
        concepts = [e.get("learning_event", {}).get("concept") for e in events if e.get("learning_event", {}).get("concept")]
        unique_concepts = list(set(concepts))
        
        if not unique_concepts:
            return {
                "current_topic": None,
                "user_goals": [],
                "preferred_learning_styles": [],
                "weak_areas": [],
                "emphasized_facts": [],
                "review_queue": [],
                "learning_history": []
            }
        
        prompt = f"""Analyze these {len(unique_concepts)} learning concepts and return ONLY JSON:
{json.dumps(unique_concepts)}

Return format:
{{"weak_areas": ["concept1", "concept2"], "current_topic": "topic", "emphasized_facts": ["fact1"]}}

Focus on identifying the 3 weakest concepts and current learning focus."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200  # Dramatically reduced
        )
        
        result = json.loads(response.choices[0].message["content"])
        
        # Ensure all required fields
        return {
            "current_topic": result.get("current_topic"),
            "user_goals": result.get("user_goals", []),
            "preferred_learning_styles": result.get("preferred_learning_styles", []),
            "weak_areas": result.get("weak_areas", [])[:5],  # Limit to 5
            "emphasized_facts": result.get("emphasized_facts", [])[:5],
            "review_queue": result.get("weak_areas", [])[:5],  # Use weak areas as review queue
            "learning_history": []
        }
        
    except Exception as e:
        print(f"❌ GPT synthesis failed: {e}")
        # Return minimal valid context
        return {
            "current_topic": None,
            "user_goals": [],
            "preferred_learning_styles": [],
            "weak_areas": [],
            "emphasized_facts": [],
            "review_queue": [],
            "learning_history": []
        }

# ------------------------------
# Background Processing
# ------------------------------

def background_synthesis(user_id: str, events: List[Dict]) -> None:
    """Background synthesis with proper locking"""
    with synthesis_locks[user_id]:
        try:
            synthesized = synthesize_context_efficient(user_id, events)
            
            # Batch update to reduce DB calls
            get_supabase().table("context_state").upsert({
                "user_id": user_id,
                "context": json.dumps(synthesized),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }, on_conflict=["user_id"]).execute()
            
            # Update cache
            context_cache[user_id] = {
                "context": synthesized,
                "timestamp": datetime.now(timezone.utc)
            }
            
            print(f"✅ Background synthesis complete for {user_id}")
            
        except Exception as e:
            print(f"❌ Background synthesis failed for {user_id}: {e}")

# ------------------------------
# API Routes
# ------------------------------

@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    """Optimized context update with intelligent synthesis"""
    
    # Extract user ID
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        user_id = user_info["sub"]
    elif update.user_id:
        user_id = update.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    # Prepare entry
    entry = update.dict(exclude={"trigger_synthesis", "user_id"})
    entry.update({
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    try:
        # Single DB insert
        get_supabase().table("context_log").insert(entry).execute()
        
        # Check if synthesis needed (non-blocking)
        if update.trigger_synthesis or update.feedback_flag:
            # Get recent events for synthesis decision
            recent_events = get_supabase().table("context_log") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("id", desc=True) \
                .limit(SYNTHESIS_THRESHOLD * 2) \
                .execute().data
            
            if should_trigger_synthesis(user_id, recent_events):
                # Launch background synthesis
                threading.Thread(
                    target=background_synthesis, 
                    args=(user_id, recent_events)
                ).start()
                
                return {"status": "ok", "synthesis_triggered": True}
        
        return {"status": "ok", "synthesis_triggered": False}
        
    except Exception as e:
        print(f"❌ Context update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update context")

@router.get("/context/cache")
def get_cached_context(user_id: str):
    """Optimized context cache with better performance"""
    now = datetime.now(timezone.utc)
    
    # Fast in-memory check
    if user_id in context_cache:
        cached = context_cache[user_id]
        age_minutes = (now - cached["timestamp"]).total_seconds() / 60
        
        if age_minutes < CACHE_TTL_MINUTES:
            return {
                "cached": True,
                "stale": False,
                "age_minutes": age_minutes,
                "context": cached["context"]
            }
    
    # Supabase lookup
    try:
        result = get_supabase().table("context_state") \
            .select("context, last_updated") \
            .eq("user_id", user_id) \
            .single() \
            .execute()
        
        if result.data:
            context = json.loads(result.data["context"])
            last_updated = datetime.fromisoformat(result.data["last_updated"].replace("Z", "+00:00"))
            age_minutes = (now - last_updated).total_seconds() / 60
            
            # Update in-memory cache
            context_cache[user_id] = {
                "context": context,
                "timestamp": now
            }
            
            return {
                "cached": True,
                "stale": age_minutes > STALE_THRESHOLD_MINUTES,
                "age_minutes": age_minutes,
                "context": context
            }
    
    except Exception as e:
        print(f"❌ Cache lookup failed: {e}")
    
    # Fallback to empty context
    empty_context = {
        "current_topic": None,
        "user_goals": [],
        "preferred_learning_styles": [],
        "weak_areas": [],
        "emphasized_facts": [],
        "review_queue": [],
        "learning_history": []
    }
    
    return {
        "cached": False,
        "stale": True,
        "age_minutes": 0,
        "context": empty_context
    }

@router.get("/context/slice")
async def get_context_slice(request: Request, focus: Optional[str] = None):
    """Smart context slice returning only relevant information"""
    
    # Extract user ID
    user_info = getattr(request.state, "user", None)
    user_id = user_info.get("sub") if user_info else request.query_params.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    try:
        # Get full context
        cache_result = get_cached_context(user_id)
        context = cache_result["context"]
        
        # Return focused slice based on request
        if focus == "review":
            return {
                "review_queue": context.get("review_queue", [])[:5],
                "weak_areas": context.get("weak_areas", [])[:5]
            }
        elif focus == "learning":
            return {
                "current_topic": context.get("current_topic"),
                "learning_history": context.get("learning_history", [])[:10]
            }
        else:
            # Default slim slice
            return {
                "current_topic": context.get("current_topic"),
                "weak_areas": context.get("weak_areas", [])[:3],
                "review_queue": context.get("review_queue", [])[:3],
                "user_goals": context.get("user_goals", [])[:3]
            }
            
    except Exception as e:
        print(f"❌ Context slice failed: {e}")
        return {
            "current_topic": None,
            "weak_areas": [],
            "review_queue": [],
            "user_goals": []
        }

@router.post("/context/reset")
def reset_context_state(request: ContextResetRequest):
    """Efficient context reset with proper cleanup"""
    user_id = request.user_id
    
    try:
        # Clear in-memory cache first
        context_cache.pop(user_id, None)
        context_cache.pop(f"synthesis_hash_{user_id}", None)
        
        # Batch database operations
        supabase = get_supabase()
        
        # Delete logs and state in parallel
        supabase.table("context_log").delete().eq("user_id", user_id).execute()
        supabase.table("context_state").delete().eq("user_id", user_id).execute()
        
        # Insert fresh context
        fresh_context = {
            "user_id": user_id,
            "context": json.dumps({
                "current_topic": None,
                "user_goals": [],
                "preferred_learning_styles": [],
                "weak_areas": [],
                "emphasized_facts": [],
                "review_queue": [],
                "learning_history": []
            }),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        supabase.table("context_state").insert(fresh_context).execute()
        
        return {"status": "context reset successfully"}
        
    except Exception as e:
        print(f"❌ Context reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

@router.get("/context/state")
def get_context_state(user_id: str):
    """Get full context state for debugging"""
    try:
        result = get_supabase().table("context_state") \
            .select("context") \
            .eq("user_id", user_id) \
            .single() \
            .execute()
        
        if result.data:
            return json.loads(result.data["context"])
        else:
            raise HTTPException(status_code=404, detail="Context not found")
            
    except Exception as e:
        print(f"❌ Context state fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch context: {e}")

@router.get("/context/logs/recent")
def get_recent_logs(user_id: str, limit: int = 10):
    """Get recent context logs for debugging"""
    try:
        result = get_supabase().table("context_log") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("id", desc=True) \
            .limit(limit) \
            .execute()
        
        return {"logs": result.data or []}
        
    except Exception as e:
        print(f"❌ Recent logs fetch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {e}")
