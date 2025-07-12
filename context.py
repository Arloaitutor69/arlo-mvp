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
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging

# ---------------------------
# Configuration
# ---------------------------
CONFIDENCE_DECAY_RATE = 0.95  # Daily decay multiplier
SYNTHESIS_THRESHOLD = 5  # Trigger synthesis every N learning events
MAX_CONTEXT_HISTORY = 50  # Concepts to keep in memory
CACHE_TTL_MINUTES = 5
STALE_THRESHOLD_MINUTES = 2
GPT_DEBOUNCE_SECONDS = 60  # Prevent GPT spam
SUPABASE_TIMEOUT_SECONDS = 3  # Fast timeout for cache endpoint
BACKGROUND_REFRESH_THRESHOLD = 10  # Minutes before triggering background refresh

# ---------------------------
# In-memory Context Cache
# ---------------------------
context_cache: Dict[str, Dict[str, Any]] = {}
synthesis_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
last_gpt_synthesis: Dict[str, datetime] = {}
executor = ThreadPoolExecutor(max_workers=10)  # For background tasks

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
# Supabase Client with Timeout
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

def supabase_with_timeout(operation_func, timeout_seconds: int = SUPABASE_TIMEOUT_SECONDS):
    """Execute Supabase operation with timeout"""
    try:
        future = executor.submit(operation_func)
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(f"Supabase operation timed out after {timeout_seconds}s")
        raise TimeoutError(f"Database operation timed out after {timeout_seconds}s")
    except Exception as e:
        logger.error(f"Supabase operation failed: {e}")
        raise

# ------------------------------
# Fast Cache Operations
# ------------------------------
def get_default_context() -> Dict[str, Any]:
    """Return default context structure"""
    return {
        "current_topic": None,
        "user_goals": [],
        "preferred_learning_styles": [],
        "weak_areas": [],
        "emphasized_facts": [],
        "review_queue": [],
        "learning_history": []
    }

def get_cached_context_fast(user_id: str) -> Dict[str, Any]:
    """Lightning-fast cache lookup - returns immediately if cached"""
    now = datetime.now(timezone.utc)
    
    # Fast in-memory check
    if user_id in context_cache:
        cached = context_cache[user_id]
        age_minutes = (now - cached["timestamp"]).total_seconds() / 60
        
        # Return immediately if cache is fresh
        if age_minutes < CACHE_TTL_MINUTES:
            return {
                "cached": True,
                "stale": False,
                "age_minutes": age_minutes,
                "context": cached["context"],
                "source": "memory_cache"
            }
        
        # If stale but not too old, return it and trigger background refresh
        if age_minutes < BACKGROUND_REFRESH_THRESHOLD:
            # Trigger background refresh
            executor.submit(background_refresh_context, user_id)
            
            return {
                "cached": True,
                "stale": True,
                "age_minutes": age_minutes,
                "context": cached["context"],
                "source": "stale_cache_refreshing"
            }
    
    # No cache or very stale - try fast DB lookup
    try:
        def db_lookup():
            supabase = get_supabase()
            result = supabase.table("context_state") \
                .select("context, last_updated") \
                .eq("user_id", user_id) \
                .limit(1) \
                .execute()
            return result.data
        
        db_result = supabase_with_timeout(db_lookup, 2)  # Very fast timeout
        
        if db_result and db_result[0]:
            row = db_result[0]
            context = json.loads(row["context"])
            
            # Handle missing last_updated gracefully
            if "last_updated" in row and row["last_updated"]:
                last_updated = datetime.fromisoformat(row["last_updated"].replace("Z", "+00:00"))
                age_minutes = (now - last_updated).total_seconds() / 60
            else:
                age_minutes = 999  # Force stale if no timestamp
            
            # Update in-memory cache
            context_cache[user_id] = {
                "context": context,
                "timestamp": now
            }
            
            return {
                "cached": True,
                "stale": age_minutes > STALE_THRESHOLD_MINUTES,
                "age_minutes": age_minutes,
                "context": context,
                "source": "database_fast"
            }
    
    except (TimeoutError, Exception) as e:
        logger.warning(f"Fast DB lookup failed for {user_id}: {e}")
        # Fall through to default
    
    # Fallback: return default context and trigger background creation
    default_context = get_default_context()
    executor.submit(ensure_user_context_exists_background, user_id)
    
    return {
        "cached": False,
        "stale": False,
        "age_minutes": 0,
        "context": default_context,
        "source": "default_creating"
    }

def background_refresh_context(user_id: str) -> None:
    """Background context refresh without blocking main thread"""
    try:
        logger.info(f"🔄 Background refreshing context for {user_id}")
        
        # Get fresh context from database
        supabase = get_supabase()
        result = supabase.table("context_state") \
            .select("context, last_updated") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        
        if result.data and result.data[0]:
            row = result.data[0]
            context = json.loads(row["context"])
            
            # Update cache
            context_cache[user_id] = {
                "context": context,
                "timestamp": datetime.now(timezone.utc)
            }
            
            logger.info(f"✅ Background refresh complete for {user_id}")
        else:
            # Context doesn't exist - create it
            ensure_user_context_exists_background(user_id)
            
    except Exception as e:
        logger.error(f"❌ Background refresh failed for {user_id}: {e}")

def ensure_user_context_exists_background(user_id: str) -> None:
    """Background context creation"""
    try:
        logger.info(f"🔧 Creating context for {user_id} in background")
        
        supabase = get_supabase()
        
        # Check if exists first
        result = supabase.table("context_state") \
            .select("context") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        
        if result.data:
            context = json.loads(result.data[0]["context"])
        else:
            # Create new context
            default_context = get_default_context()
            
            new_row = {
                "user_id": user_id,
                "context": json.dumps(default_context),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            supabase.table("context_state").insert(new_row).execute()
            context = default_context
            logger.info(f"✅ Created new context for user {user_id}")
        
        # Update cache
        context_cache[user_id] = {
            "context": context,
            "timestamp": datetime.now(timezone.utc)
        }
        
    except Exception as e:
        logger.error(f"❌ Background context creation failed for {user_id}: {e}")

# ------------------------------
# Schema Management
# ------------------------------
def ensure_schema_exists():
    """Ensure required database schema exists"""
    try:
        supabase = get_supabase()
        
        # Check if context_state has last_updated column
        result = supabase.table("context_state").select("last_updated").limit(1).execute()
        
    except Exception as e:
        if "last_updated" in str(e):
            logger.warning("⚠️  Missing last_updated column in context_state table")
            logger.info("🔧 Run this SQL in Supabase:")
            logger.info("ALTER TABLE context_state ADD COLUMN last_updated TIMESTAMPTZ DEFAULT NOW();")
            logger.info("UPDATE context_state SET last_updated = NOW() WHERE last_updated IS NULL;")
        raise e

def ensure_user_context_exists(user_id: str) -> Dict[str, Any]:
    """Ensure user has a context_state row, create if missing"""
    try:
        supabase = get_supabase()
        
        # Try to get existing context
        result = supabase.table("context_state").select("*").eq("user_id", user_id).execute()
        
        if result.data:
            return json.loads(result.data[0]["context"])
        
        # Create new context if missing
        default_context = get_default_context()
        
        new_row = {
            "user_id": user_id,
            "context": json.dumps(default_context),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        supabase.table("context_state").insert(new_row).execute()
        logger.info(f"✅ Created new context for user {user_id}")
        return default_context
        
    except Exception as e:
        logger.error(f"❌ Error ensuring user context: {e}")
        # Return default context as fallback
        return get_default_context()

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
# Validation and Sanitization
# ------------------------------
def validate_and_clean_event(event: Any) -> Optional[Dict[str, Any]]:
    """Validate and clean event data before processing"""
    if not event or not isinstance(event, dict):
        return None
    
    # Ensure timestamp exists and is valid
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Validate timestamp format
    try:
        datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    except:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Ensure learning_event is valid
    if "learning_event" in event and event["learning_event"]:
        le = event["learning_event"]
        if not isinstance(le, dict) or not le.get("concept"):
            event["learning_event"] = None
    
    return event

def sanitize_context_update(update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean update dict before database insertion"""
    # Remove None values and empty strings
    cleaned = {}
    for key, value in update_dict.items():
        if value is not None and value != "":
            if isinstance(value, list) and not value:
                continue  # Skip empty lists
            cleaned[key] = value
    
    return cleaned

# ------------------------------
# Core Memory Functions
# ------------------------------
def aggregate_learning_events(events: List[Dict]) -> Dict[str, ConceptMemory]:
    """Efficiently aggregate learning events by concept with validation"""
    concept_map: Dict[str, ConceptMemory] = {}
    
    for event in events:
        # Validate event first
        clean_event = validate_and_clean_event(event)
        if not clean_event:
            continue
            
        le = clean_event.get("learning_event")
        if not le or not isinstance(le, dict) or not le.get("concept"):
            continue
            
        concept = le["concept"]
        confidence = le.get("confidence", 0.5)
        depth = le.get("depth", "shallow")
        source = clean_event.get("source", "unknown")
        
        try:
            timestamp = datetime.fromisoformat(clean_event["timestamp"].replace("Z", "+00:00"))
        except:
            timestamp = datetime.now(timezone.utc)
        
        if concept in concept_map:
            # Update existing concept
            memory = concept_map[concept]
            memory.confidence = max(memory.confidence, confidence)
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
    
    return sorted(due_for_review, key=lambda c: concepts[c].next_review or now)

def create_context_hash(events: List[Dict]) -> str:
    """Create hash to detect if synthesis is needed"""
    try:
        # Only hash the essential parts to avoid noise
        essential_data = []
        for event in events:
            if event and isinstance(event, dict) and event.get("learning_event"):
                essential_data.append(event["learning_event"].get("concept", ""))
        
        content = json.dumps(essential_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        logger.error(f"❌ Hash creation failed: {e}")
        return str(hash(str(events)))

def should_trigger_synthesis(user_id: str, events: List[Dict]) -> bool:
    """Smart synthesis triggering with debouncing"""
    # Check if enough new events
    valid_events = [e for e in events if validate_and_clean_event(e)]
    if len(valid_events) < SYNTHESIS_THRESHOLD:
        return False
    
    # Check GPT debounce
    now = datetime.now()
    if user_id in last_gpt_synthesis:
        time_since_last = (now - last_gpt_synthesis[user_id]).total_seconds()
        if time_since_last < GPT_DEBOUNCE_SECONDS:
            return False
    
    # Check if content has changed significantly
    current_hash = create_context_hash(valid_events)
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
        # Validate all events first
        valid_events = [validate_and_clean_event(e) for e in recent_events]
        valid_events = [e for e in valid_events if e]  # Remove None values
        
        if not valid_events:
            logger.warning("⚠️  No valid events for synthesis")
            return ensure_user_context_exists(user_id)
        
        # Aggregate learning events efficiently
        concepts = aggregate_learning_events(valid_events)
        
        # Extract metadata from most recent events
        meta_fields = {
            "current_topic": None,
            "user_goals": [],
            "preferred_learning_styles": [],
            "emphasized_facts": []
        }
        
        for event in reversed(valid_events):  # Most recent first
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
        logger.error(f"❌ Efficient synthesis failed: {e}")
        return fallback_to_gpt_synthesis(user_id, recent_events)

def fallback_to_gpt_synthesis(user_id: str, events: List[Dict]) -> Dict[str, Any]:
    """GPT synthesis as fallback for complex cases with debouncing"""
    try:
        # Check debounce
        now = datetime.now()
        if user_id in last_gpt_synthesis:
            time_since_last = (now - last_gpt_synthesis[user_id]).total_seconds()
            if time_since_last < GPT_DEBOUNCE_SECONDS:
                logger.warning(f"⚠️  GPT synthesis debounced for {user_id}")
                return ensure_user_context_exists(user_id)
        
        # Update debounce timestamp
        last_gpt_synthesis[user_id] = now
        
        # Validate and extract concepts
        valid_events = [validate_and_clean_event(e) for e in events]
        valid_events = [e for e in valid_events if e]
        
        concepts = []
        for event in valid_events:
            le = event.get("learning_event")
            if le and isinstance(le, dict) and le.get("concept"):
                concepts.append(le["concept"])
        
        unique_concepts = list(set(concepts))
        
        if not unique_concepts:
            logger.warning("⚠️  No concepts found for GPT synthesis")
            return ensure_user_context_exists(user_id)
        
        prompt = f"""Analyze these {len(unique_concepts)} learning concepts and return ONLY JSON:
{json.dumps(unique_concepts[:20])}

Return format:
{{"weak_areas": ["concept1", "concept2"], "current_topic": "topic", "emphasized_facts": ["fact1"]}}

Focus on identifying the 3 weakest concepts and current learning focus."""

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        result = json.loads(response.choices[0].message["content"])
        
        # Ensure all required fields
        return {
            "current_topic": result.get("current_topic"),
            "user_goals": result.get("user_goals", []),
            "preferred_learning_styles": result.get("preferred_learning_styles", []),
            "weak_areas": result.get("weak_areas", [])[:5],
            "emphasized_facts": result.get("emphasized_facts", [])[:5],
            "review_queue": result.get("weak_areas", [])[:5],
            "learning_history": []
        }
        
    except Exception as e:
        logger.error(f"❌ GPT synthesis failed: {e}")
        # Return user's existing context or default
        return ensure_user_context_exists(user_id)

# ------------------------------
# Background Processing
# ------------------------------
def background_synthesis(user_id: str, events: List[Dict]) -> None:
    """Background synthesis with proper locking and error handling"""
    with synthesis_locks[user_id]:
        try:
            logger.info(f"🔄 Starting background synthesis for {user_id}")
            
            synthesized = synthesize_context_efficient(user_id, events)
            
            # Use upsert with proper error handling
            supabase = get_supabase()
            
            update_data = {
                "user_id": user_id,
                "context": json.dumps(synthesized),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            result = supabase.table("context_state").upsert(
                update_data, 
                on_conflict="user_id"
            ).execute()
            
            logger.info(f"✅ Background synthesis complete for {user_id}")
            logger.info(f"📊 Synthesis result: {len(synthesized.get('learning_history', []))} concepts")
            
            # Update cache
            context_cache[user_id] = {
                "context": synthesized,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"❌ Background synthesis failed for {user_id}: {e}")
            # Ensure user has some context even if synthesis fails
            ensure_user_context_exists(user_id)

# ------------------------------
# API Routes
# ------------------------------
@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    """Optimized context update with robust error handling"""
    
    # Extract user ID
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        user_id = user_info["sub"]
    elif update.user_id:
        user_id = update.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    # Prepare and sanitize entry
    entry_dict = update.dict(exclude={"trigger_synthesis", "user_id"})
    entry_dict.update({
        "user_id": user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Clean the entry
    entry = sanitize_context_update(entry_dict)
    
    try:
        # Ensure user context exists (background task)
        executor.submit(ensure_user_context_exists_background, user_id)
        
        # Insert with detailed logging
        def insert_log():
            supabase = get_supabase()
            return supabase.table("context_log").insert(entry).execute()
        
        # Try to insert with timeout
        try:
            result = supabase_with_timeout(insert_log, 5)
            logger.info(f"✅ Context log inserted for {user_id}")
        except TimeoutError:
            logger.warning(f"⚠️ Context log insert timed out for {user_id}, queuing for background")
            # Queue for background processing
            executor.submit(insert_log)
            return {"status": "ok", "synthesis_triggered": False, "queued": True}
        
        # Check if synthesis needed (non-blocking)
        if update.trigger_synthesis or update.feedback_flag:
            # Get recent events for synthesis decision
            def get_recent_events():
                supabase = get_supabase()
                return supabase.table("context_log") \
                    .select("*") \
                    .eq("user_id", user_id) \
                    .order("id", desc=True) \
                    .limit(SYNTHESIS_THRESHOLD * 2) \
                    .execute().data
            
            try:
                recent_events = supabase_with_timeout(get_recent_events, 3)
                
                if should_trigger_synthesis(user_id, recent_events):
                    # Launch background synthesis
                    executor.submit(background_synthesis, user_id, recent_events)
                    return {"status": "ok", "synthesis_triggered": True}
            except TimeoutError:
                logger.warning(f"⚠️ Recent events query timed out for {user_id}")
                # Still return success, synthesis will happen later
        
        return {"status": "ok", "synthesis_triggered": False}
        
    except Exception as e:
        logger.error(f"❌ Context update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update context: {str(e)}")

@router.get("/context/cache")
def get_context_cache(user_id: str):
    """Ultra-fast context cache - optimized for speed"""
    start_time = time.time()
    
    try:
        result = get_cached_context_fast(user_id)
        
        # Add timing info
        elapsed_ms = (time.time() - start_time) * 1000
        result["response_time_ms"] = round(elapsed_ms, 2)
        
        logger.info(f"📊 Context cache for {user_id}: {result['source']} in {elapsed_ms:.2f}ms")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Context cache failed for {user_id}: {e}")
        
        # Emergency fallback
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "cached": False,
            "stale": True,
            "age_minutes": 0,
            "context": get_default_context(),
            "source": "emergency_fallback",
            "response_time_ms": round(elapsed_ms, 2),
            "error": str(e)
        }

@router.get("/context/slice")
async def get_context_slice(request: Request, focus: Optional[str] = None):
    """Smart context slice with robust error handling"""
    
    # Extract user ID
    user_info = getattr(request.state, "user", None)
    user_id = user_info.get("sub") if user_info else request.query_params.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    try:
        # Get full context using fast cache
        cache_result = get_cached_context_fast(user_id)
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
        logger.error(f"❌ Context slice failed: {e}")
        return {
            "current"
        }
