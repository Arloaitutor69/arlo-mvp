from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict, Any
from datetime import datetime, timedelta, timezone
import json
from openai import OpenAI
import os
from supabase import create_client, Client
import requests
from collections import defaultdict
import threading
import asyncio
import hashlib
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
CONFIDENCE_DECAY_RATE = 0.95
SYNTHESIS_THRESHOLD = 5
MAX_CONTEXT_HISTORY = 50
CACHE_TTL_MINUTES = 5
STALE_THRESHOLD_MINUTES = 2
GPT_DEBOUNCE_SECONDS = 60
SUPABASE_TIMEOUT_SECONDS = 3
BACKGROUND_REFRESH_THRESHOLD = 10

# Global state
context_cache: Dict[str, Dict[str, Any]] = {}
synthesis_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
last_gpt_synthesis: Dict[str, datetime] = {}
executor = ThreadPoolExecutor(max_workers=10)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConceptMemory:
    concept: str
    confidence: float
    depth: str
    last_seen: datetime
    repetition_count: int
    sources: List[str]
    next_review: Optional[datetime] = None
    
    def calculate_retention(self) -> float:
        days_since = (datetime.now() - self.last_seen).days
        return self.confidence * (CONFIDENCE_DECAY_RATE ** days_since)
    
    def schedule_review(self) -> None:
        interval_days = min(2 ** self.repetition_count, 30)
        self.next_review = datetime.now() + timedelta(days=interval_days)

# Supabase client
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
    try:
        future = executor.submit(operation_func)
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        logger.warning(f"Supabase operation timed out after {timeout_seconds}s")
        raise TimeoutError(f"Database operation timed out after {timeout_seconds}s")
    except Exception as e:
        logger.error(f"Supabase operation failed: {e}")
        raise

def get_default_context() -> Dict[str, Any]:
    return {
        "current_topic": None,
        "user_goals": [],
        "preferred_learning_styles": [],
        "weak_areas": [],
        "emphasized_facts": [],
        "review_queue": [],
        "learning_history": []
    }

def clear_user_cache(user_id: str) -> None:
    if user_id in context_cache:
        del context_cache[user_id]
        logger.info(f"Cleared in-memory cache for {user_id}")

def get_cached_context_fast(user_id: str) -> Dict[str, Any]:
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
                "context": cached["context"],
                "source": "memory_cache"
            }
        
        if age_minutes < BACKGROUND_REFRESH_THRESHOLD:
            executor.submit(background_refresh_context, user_id)
            return {
                "cached": True,
                "stale": True,
                "age_minutes": age_minutes,
                "context": cached["context"],
                "source": "stale_cache_refreshing"
            }
    
    # Try fast DB lookup
    try:
        def db_lookup():
            supabase = get_supabase()
            result = supabase.table("context_state") \
                .select("context, last_updated") \
                .eq("user_id", user_id) \
                .limit(1) \
                .execute()
            return result.data
        
        db_result = supabase_with_timeout(db_lookup, 2)
        
        if db_result and db_result[0]:
            row = db_result[0]
            context = json.loads(row["context"])
            
            if "last_updated" in row and row["last_updated"]:
                last_updated = datetime.fromisoformat(row["last_updated"].replace("Z", "+00:00"))
                age_minutes = (now - last_updated).total_seconds() / 60
            else:
                age_minutes = 999
            
            context_cache[user_id] = {"context": context, "timestamp": now}
            
            return {
                "cached": True,
                "stale": age_minutes > STALE_THRESHOLD_MINUTES,
                "age_minutes": age_minutes,
                "context": context,
                "source": "database_fast"
            }
    
    except (TimeoutError, Exception) as e:
        logger.warning(f"Fast DB lookup failed for {user_id}: {e}")
    
    # Fallback: return default and create in background
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
    try:
        logger.info(f"Background refreshing context for {user_id}")
        
        supabase = get_supabase()
        result = supabase.table("context_state") \
            .select("context, last_updated") \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()
        
        if result.data and result.data[0]:
            row = result.data[0]
            context = json.loads(row["context"])
            context_cache[user_id] = {"context": context, "timestamp": datetime.now(timezone.utc)}
            logger.info(f"Background refresh complete for {user_id}")
        else:
            ensure_user_context_exists_background(user_id)
            
    except Exception as e:
        logger.error(f"Background refresh failed for {user_id}: {e}")

def ensure_user_context_exists_background(user_id: str) -> None:
    try:
        logger.info(f"Creating context for {user_id} in background")
        
        supabase = get_supabase()
        result = supabase.table("context_state").select("context").eq("user_id", user_id).limit(1).execute()
        
        if result.data:
            context = json.loads(result.data[0]["context"])
        else:
            default_context = get_default_context()
            new_row = {
                "user_id": user_id,
                "context": json.dumps(default_context),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            supabase.table("context_state").insert(new_row).execute()
            context = default_context
            logger.info(f"Created new context for user {user_id}")
        
        context_cache[user_id] = {"context": context, "timestamp": datetime.now(timezone.utc)}
        
    except Exception as e:
        logger.error(f"Background context creation failed for {user_id}: {e}")

def ensure_user_context_exists(user_id: str) -> Dict[str, Any]:
    try:
        supabase = get_supabase()
        result = supabase.table("context_state").select("*").eq("user_id", user_id).execute()
        
        if result.data:
            return json.loads(result.data[0]["context"])
        
        default_context = get_default_context()
        new_row = {
            "user_id": user_id,
            "context": json.dumps(default_context),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("context_state").insert(new_row).execute()
        logger.info(f"Created new context for user {user_id}")
        return default_context
        
    except Exception as e:
        logger.error(f"Error ensuring user context: {e}")
        return get_default_context()

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

# GPT-5-nano JSON Schema for synthesis
SYNTHESIS_SCHEMA = {
    "name": "context_synthesis",
    "schema": {
        "type": "object",
        "strict": True,
        "properties": {
            "weak_areas": {
                "type": "array",
                "maxItems": 5,
                "items": {"type": "string", "minLength": 1}
            },
            "current_topic": {
                "type": "string"
            },
            "emphasized_facts": {
                "type": "array", 
                "maxItems": 5,
                "items": {"type": "string", "minLength": 1}
            }
        },
        "required": ["weak_areas", "current_topic", "emphasized_facts"],
        "additionalProperties": False
    }
}

def validate_and_clean_event(event: Any) -> Optional[Dict[str, Any]]:
    if not event or not isinstance(event, dict):
        return None
    
    if "timestamp" not in event:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    try:
        datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
    except:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    if "learning_event" in event and event["learning_event"]:
        le = event["learning_event"]
        if not isinstance(le, dict) or not le.get("concept"):
            event["learning_event"] = None
    
    return event

def sanitize_context_update(update_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in update_dict.items():
        if value is not None and value != "":
            if isinstance(value, list) and not value:
                continue
            cleaned[key] = value
    return cleaned

def aggregate_learning_events(events: List[Dict]) -> Dict[str, ConceptMemory]:
    concept_map: Dict[str, ConceptMemory] = {}
    
    for event in events:
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
            memory = concept_map[concept]
            memory.confidence = max(memory.confidence, confidence)
            memory.depth = max(memory.depth, depth, key=lambda x: ["shallow", "intermediate", "deep"].index(x))
            memory.last_seen = max(memory.last_seen, timestamp)
            memory.repetition_count += 1
            if source not in memory.sources:
                memory.sources.append(source)
        else:
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
    weak_concepts = []
    for concept, memory in concepts.items():
        current_retention = memory.calculate_retention()
        if current_retention < threshold:
            weak_concepts.append(concept)
    return sorted(weak_concepts, key=lambda c: concepts[c].calculate_retention())

def generate_review_queue(concepts: Dict[str, ConceptMemory]) -> List[str]:
    now = datetime.now()
    due_for_review = []
    
    for concept, memory in concepts.items():
        if memory.next_review is None:
            memory.schedule_review()
        
        if memory.next_review <= now:
            due_for_review.append(concept)
    
    return sorted(due_for_review, key=lambda c: concepts[c].next_review or now)

def create_context_hash(events: List[Dict]) -> str:
    try:
        essential_data = []
        for event in events:
            if event and isinstance(event, dict) and event.get("learning_event"):
                essential_data.append(event["learning_event"].get("concept", ""))
        
        content = json.dumps(essential_data, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Hash creation failed: {e}")
        return str(hash(str(events)))

def should_trigger_synthesis(user_id: str, events: List[Dict]) -> bool:
    valid_events = [e for e in events if validate_and_clean_event(e)]
    if len(valid_events) < SYNTHESIS_THRESHOLD:
        return False
    
    now = datetime.now()
    if user_id in last_gpt_synthesis:
        time_since_last = (now - last_gpt_synthesis[user_id]).total_seconds()
        if time_since_last < GPT_DEBOUNCE_SECONDS:
            return False
    
    current_hash = create_context_hash(valid_events)
    cache_key = f"synthesis_hash_{user_id}"
    
    if cache_key in context_cache:
        last_hash = context_cache[cache_key].get("hash")
        if current_hash == last_hash:
            return False
    
    context_cache[cache_key] = {"hash": current_hash}
    return True

def synthesize_context_efficient(user_id: str, recent_events: List[Dict]) -> Dict[str, Any]:
    try:
        valid_events = [validate_and_clean_event(e) for e in recent_events]
        valid_events = [e for e in valid_events if e]
        
        if not valid_events:
            logger.warning("No valid events for synthesis")
            return ensure_user_context_exists(user_id)
        
        concepts = aggregate_learning_events(valid_events)
        
        # Extract metadata from recent events
        meta_fields = {"current_topic": None, "user_goals": [], "preferred_learning_styles": [], "emphasized_facts": []}
        
        for event in reversed(valid_events):
            for field in meta_fields:
                if event.get(field) and not meta_fields[field]:
                    meta_fields[field] = event[field]
        
        weak_areas = identify_weak_areas(concepts)
        review_queue = generate_review_queue(concepts)
        
        learning_history = []
        for concept, memory in concepts.items():
            learning_history.append({
                "concept": memory.concept,
                "confidence": memory.calculate_retention(),
                "depth": memory.depth,
                "repetition_count": memory.repetition_count,
                "sources": memory.sources[:3],
                "last_seen": memory.last_seen.isoformat(),
                "next_review": memory.next_review.isoformat() if memory.next_review else None
            })
        
        learning_history.sort(key=lambda x: (datetime.fromisoformat(x["last_seen"]), x["confidence"]), reverse=True)
        
        return {
            **meta_fields,
            "weak_areas": weak_areas[:10],
            "review_queue": review_queue[:10],
            "learning_history": learning_history[:MAX_CONTEXT_HISTORY]
        }
        
    except Exception as e:
        logger.error(f"Efficient synthesis failed: {e}")
        return fallback_to_gpt_synthesis(user_id, recent_events)

def fallback_to_gpt_synthesis(user_id: str, events: List[Dict]) -> Dict[str, Any]:
    try:
        now = datetime.now()
        if user_id in last_gpt_synthesis:
            time_since_last = (now - last_gpt_synthesis[user_id]).total_seconds()
            if time_since_last < GPT_DEBOUNCE_SECONDS:
                logger.warning(f"GPT synthesis debounced for {user_id}")
                return ensure_user_context_exists(user_id)
        
        last_gpt_synthesis[user_id] = now
        
        valid_events = [validate_and_clean_event(e) for e in events]
        valid_events = [e for e in valid_events if e]
        
        concepts = []
        for event in valid_events:
            le = event.get("learning_event")
            if le and isinstance(le, dict) and le.get("concept"):
                concepts.append(le["concept"])
        
        unique_concepts = list(set(concepts))
        
        if not unique_concepts:
            logger.warning("No concepts found for GPT synthesis")
            return ensure_user_context_exists(user_id)
        
        user_prompt = f"""Analyze these {len(unique_concepts)} learning concepts and identify weak areas and current focus:
{json.dumps(unique_concepts[:20])}

Focus on identifying the 3 weakest concepts and current learning topic."""

        messages = [
            {"role": "system", "content": "You are a learning analytics expert. Analyze student concepts and identify weak areas and learning focus."},
            {"role": "user", "content": user_prompt}
        ]
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": SYNTHESIS_SCHEMA
            },
            reasoning_effort="low"
        )
        
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        
        return {
            "current_topic": result.get("current_topic"),
            "user_goals": [],
            "preferred_learning_styles": [],
            "weak_areas": result.get("weak_areas", []),
            "emphasized_facts": result.get("emphasized_facts", []),
            "review_queue": result.get("weak_areas", []),
            "learning_history": []
        }
        
    except Exception as e:
        logger.error(f"GPT synthesis failed: {e}")
        return ensure_user_context_exists(user_id)

def background_synthesis(user_id: str, events: List[Dict]) -> None:
    with synthesis_locks[user_id]:
        try:
            logger.info(f"Starting background synthesis for {user_id}")
            
            synthesized = synthesize_context_efficient(user_id, events)
            
            supabase = get_supabase()
            update_data = {
                "user_id": user_id,
                "context": json.dumps(synthesized),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            result = supabase.table("context_state").upsert(update_data, on_conflict="user_id").execute()
            
            logger.info(f"Background synthesis complete for {user_id}")
            logger.info(f"Synthesis result: {len(synthesized.get('learning_history', []))} concepts")
            
            context_cache[user_id] = {"context": synthesized, "timestamp": datetime.now(timezone.utc)}
            
        except Exception as e:
            logger.error(f"Background synthesis failed for {user_id}: {e}")
            ensure_user_context_exists(user_id)

@router.post("/context/update")
async def update_context(update: ContextUpdate, request: Request):
    # Extract user ID
    user_info = getattr(request.state, "user", None)
    if user_info and "sub" in user_info:
        user_id = user_info["sub"]
    elif update.user_id:
        user_id = update.user_id
    else:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    entry_dict = update.dict(exclude={"trigger_synthesis", "user_id"})
    entry_dict.update({"user_id": user_id, "timestamp": datetime.now(timezone.utc).isoformat()})
    entry = sanitize_context_update(entry_dict)
    
    try:
        executor.submit(ensure_user_context_exists_background, user_id)
        
        def insert_log():
            supabase = get_supabase()
            return supabase.table("context_log").insert(entry).execute()
        
        try:
            result = supabase_with_timeout(insert_log, 5)
            logger.info(f"Context log inserted for {user_id}")
        except TimeoutError:
            logger.warning(f"Context log insert timed out for {user_id}, queuing for background")
            executor.submit(insert_log)
            return {"status": "ok", "synthesis_triggered": False, "queued": True}
        
        if update.trigger_synthesis or update.feedback_flag:
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
                    executor.submit(background_synthesis, user_id, recent_events)
                    return {"status": "ok", "synthesis_triggered": True}
            except TimeoutError:
                logger.warning(f"Recent events query timed out for {user_id}")
        
        return {"status": "ok", "synthesis_triggered": False}
        
    except Exception as e:
        logger.error(f"Context update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update context: {str(e)}")

@router.get("/context/cache")
def get_context_cache(user_id: str):
    start_time = time.time()
    
    try:
        result = get_cached_context_fast(user_id)
        elapsed_ms = (time.time() - start_time) * 1000
        result["response_time_ms"] = round(elapsed_ms, 2)
        logger.info(f"Context cache for {user_id}: {result['source']} in {elapsed_ms:.2f}ms")
        return result
        
    except Exception as e:
        logger.error(f"Context cache failed for {user_id}: {e}")
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
    user_info = getattr(request.state, "user", None)
    user_id = user_info.get("sub") if user_info else request.query_params.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user ID")
    
    try:
        cache_result = get_cached_context_fast(user_id)
        context = cache_result["context"]
        
        if focus == "review":
            slice_data = {
                "review_queue": context.get("review_queue", [])[:5],
                "weak_areas": context.get("weak_areas", [])[:5]
            }
        elif focus == "learning":
            slice_data = {
                "current_topic": context.get("current_topic"),
                "learning_history": context.get("learning_history", [])[:10]
            }
        else:
            slice_data = {
                "current_topic": context.get("current_topic"),
                "weak_areas": context.get("weak_areas", [])[:3],
                "review_queue": context.get("review_queue", [])[:3],
                "emphasized_facts": context.get("emphasized_facts", [])[:3]
            }
        
        return {
            **slice_data,
            "_cache_info": {
                "source": cache_result.get("source"),
                "age_minutes": cache_result.get("age_minutes"),
                "stale": cache_result.get("stale", False)
            }
        }
        
    except Exception as e:
        logger.error(f"Context slice failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get context slice: {str(e)}")

@router.post("/context/reset")
async def reset_context_state(request: ContextResetRequest):
    user_id = request.user_id
    
    try:
        logger.info(f"Starting context reset for {user_id}")
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE")
        
        if not supabase_url or not supabase_key:
            raise HTTPException(status_code=500, detail="Missing Supabase environment variables")
        
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json"
        }
        
        # Clear cache first
        clear_user_cache(user_id)
        
        # Delete context logs and state
        try:
            delete_logs_url = f"{supabase_url}/rest/v1/context_log?user_id=eq.{user_id}"
            delete_res = requests.delete(delete_logs_url, headers=headers, timeout=10)
            delete_res.raise_for_status()
            logger.info(f"Deleted context_log entries for {user_id}")
        except Exception as e:
            logger.error(f"Failed to delete context_log for {user_id}: {e}")
        
        try:
            delete_ctx_url = f"{supabase_url}/rest/v1/context_state?user_id=eq.{user_id}"
            delete_ctx_res = requests.delete(delete_ctx_url, headers=headers, timeout=10)
            delete_ctx_res.raise_for_status()
            logger.info(f"Deleted context_state for {user_id}")
        except Exception as e:
            logger.warning(f"Failed to delete context_state for {user_id}: {e}")
        
        # Insert fresh context
        reset_context = {
            "user_id": user_id,
            "context": json.dumps(get_default_context()),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        reset_res = requests.post(f"{supabase_url}/rest/v1/context_state", json=reset_context, headers=headers, timeout=10)
        reset_res.raise_for_status()
        logger.info(f"Created fresh context_state for {user_id}")
        
        # Clear locks and update cache
        if user_id in synthesis_locks:
            del synthesis_locks[user_id]
        if user_id in last_gpt_synthesis:
            del last_gpt_synthesis[user_id]
        
        context_cache[user_id] = {"context": get_default_context(), "timestamp": datetime.now(timezone.utc)}
        
        logger.info(f"Context reset complete for {user_id}")
        
        return {
            "status": "context fully reset",
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "All context data cleared and reset to default state"
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed during reset for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database operation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Context reset failed for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")
