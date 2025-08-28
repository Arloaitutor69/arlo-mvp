from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import os
from openai import OpenAI
import logging
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Environment Variables --- #
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Input Schema --- #
class CombinedRequest(BaseModel):
    teaching_description: str  # Main teaching content description
    youtube_topic: str  # Specific YouTube search topic
    subject: Optional[str] = None
    level: Optional[str] = "intermediate"
    test_type: Optional[str] = None
    max_duration_minutes: Optional[int] = 30

# --- Output Schemas --- #
class TeachingBlock(BaseModel):
    title: str
    content: str

class VideoSegment(BaseModel):
    start_time: str
    end_time: str
    topic: str
    relevance_score: float

class YouTubeVideo(BaseModel):
    title: str
    video_id: str
    url: str
    thumbnail: str
    duration: str
    channel_title: str
    view_count: int
    published_at: str
    query_used: str
    relevant_segments: List[VideoSegment]
    overall_relevance_score: float

class CombinedResponse(BaseModel):
    lesson: List[TeachingBlock]
    youtube_video: Optional[YouTubeVideo] = None
    status: str

# --- GPT System Prompt for Teaching --- #
GPT_SYSTEM_PROMPT = """You are an expert tutor who excels in teaching difficult content in a way that is engaging and the most simple easy to understand way possible.
Create exactly 8-14 teaching blocks that thoroughly cover ALL aspects of the requested topic. Your explanations should sound like you're talking directly to the student, never like a textbook. 

CRITICAL STYLE REQUIREMENTS:
- MOST IMPORTANT: mimick exactly the assistant examples, particularly the casual easy to understand nature of explenations with lots of examples and clarifications. 
- Always use **simple words** and explain technical terms in plain English the first time they appear.
- Always include **relatable analogies, examples, or metaphors**
- Always keep a **conversational tone**: ask rhetorical questions, say "think of it like…" or "imagine…".
- Never drift into formal research paper or lecture style.
- Never introduce advanced words without breaking them down.
- Never output bullet lists without adding a quick analogy or everyday example to ground them.
- structure each lesson in accesible way with bullet point breakdowns when helpful 

CRITICAL REQUIREMENTS:
1. Return ONLY JSON data conforming to the schema, never the schema itself.
2. Output ONLY valid JSON format with proper escaping.
3. Use double quotes, escape internal quotes as \\\".
4. Use \\n for line breaks within content.
5. No trailing commas.

TEACHING BLOCK STRUCTURE:
- Each block should fully explain 1-2 subtopics in an easy to understand way.
- Cover all aspects of the requested topic comprehensively.
- Use bullet points with * for key concepts and lists.
- Use **bold formatting** for important terms and concepts.
- Include examples in parentheses when helpful.

CONTENT QUALITY STANDARDS:
- Each block should be ~70-130 words.
- ONLY MENTION information relevant to a test, not tangential information.
- Define all technical terms at first mention and assume student has almost zero prior knowledge

--- Most Important ---
1. Always output exactly 8-14 separate teaching blocks.
2. Mimic teaching style of examples as closely as possible, use same casual language, structure, and explanation style.
"""

# --- Teaching Response Schema --- #
class TeachingResponse(BaseModel):
    lesson: List[TeachingBlock]

# --- JSON examples --- #
ASSISTANT_EXAMPLE_JSON_1 = """
{
  "lesson": [
    {
      "title": "What is Economics?",
      "content": "Economics is the study of how people make choices about their limited resources. Everyone—individuals, businesses, and governments—has to make decisions about what to use, what to save, and what to trade.\\n\\n**Key ideas:**\\n* **Scarcity:** Resources (money, time, food, etc.) are limited. We can\\'t have everything we want.\\n* **Choices:** Because of scarcity, we make decisions about what to use resources for.\\n* **Opportunity Cost:** Whenever you choose one thing, you give up the next best alternative. (Example: if you spend $10 on lunch, you can\\'t spend that $10 on a movie ticket.)\\n\\nSo economics is the study of **who gets what, how they can get it, and why!**"
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_2 = """
{
  "lesson": [
    {
      "title": "The Cell Membrane: Your Cell's Security System",
      "content": "The **cell membrane** works like a security guard or a bouncer at a door. It decides what can come into the cell and what has to stay out.\\n\\n**Key things to know:**\\n* It's made of a double layer of phospholipids (kind of like a thin soapy bubble wall)\\n* It is **selectively permeable** – a fancy term for deciding what goes in and what comes out\\n* It has special **transport proteins** that act like doors or ID checkers for bigger molecules when they want to enter or leave\\n\\n**What actually gets through:**\\n* Water and very small molecules can slip in and out easily\\n* Larger molecules need a special 'door' (transport proteins)\\n* Waste gets pushed out so the cell stays clean"
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_3 = """
{
  "lesson": [
    {
      "title": "Micro vs. Macro Economics",
      "content": "Economics is split into two main 'worlds.'\\n\\n**Microeconomics:** The study of small, individual decisions.\\n* Example: A family choosing whether to eat out or cook at home\\n* Example: A business deciding how much to charge for sneakers\\n\\n**Macroeconomics:** The study of the whole economy.\\n* Example: Why is inflation rising?\\n* Example: Why do some countries grow richer while others struggle?\\n\\nThink of it like zooming in with a camera: **Micro = close-up**, **Macro = wide-angle** view of the entire economy."
    }
  ]
}
"""

ASSISTANT_EXAMPLE_JSON_4 = """
{
  "lesson": [
    {
      "title": "Cells and Cell Theory",
      "content": "A **cell** is the smallest living piece of life that can do all the important things like grow, use energy, react to surroundings, and reproduce.\\n\\n**Cell Theory says:**\\n* All living things are made of cells\\n* All cells come from other cells\\n\\n**Types of cells:**\\n* **Prokaryotes:** Simple cells with no nucleus, DNA floats in cytoplasm, reproduce fast by binary fission (split in two)\\n* **Eukaryotes:** Found in plants and animals, more complex with a nucleus to protect DNA, like miniature cities with factories and workers\\n\\nCells often team up to make bigger organisms (like humans with trillions of cells working together)."
    }
  ]
}
"""

# --- YouTube Helper Functions --- #
def extract_keywords_simple(youtube_topic: str, subject: Optional[str]) -> List[str]:
    """Extract keywords from YouTube topic and subject."""
    keywords = []
    
    if subject:
        keywords.append(subject)
    
    if youtube_topic:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', youtube_topic.lower())
        meaningful_words = [word for word in words if word not in stop_words]
        
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords.extend([word for word, freq in sorted_words[:5]])
    
    return keywords[:5] if keywords else ["tutorial", "lesson"]

def generate_search_query(youtube_topic: str, subject: Optional[str], level: str) -> str:
    """Generate a single optimized search query since topic is pre-specified."""
    query_parts = []
    
    if subject:
        query_parts.append(subject)
    
    # Add the specific YouTube topic
    query_parts.append(youtube_topic)
    
    # Add level-specific terms
    if level == "beginner":
        query_parts.append("basics tutorial")
    elif level == "advanced":
        query_parts.append("advanced")
    else:
        query_parts.append("explained")
    
    return " ".join(query_parts)[:80]

def parse_duration_to_seconds(duration: str) -> int:
    """Convert YouTube duration format (PT1H2M3S) to seconds."""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds

def format_duration(seconds: int) -> str:
    """Convert seconds to MM:SS or HH:MM:SS format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def create_default_segments(youtube_topic: str, duration_seconds: int) -> List[VideoSegment]:
    """Create reasonable default segments based on video duration."""
    segments = []
    topic = youtube_topic if youtube_topic else "Main content"
    
    if duration_seconds <= 300:  # 5 minutes or less
        segments.append(VideoSegment(
            start_time="0:30",
            end_time=format_duration(min(duration_seconds - 10, 240)),
            topic=topic,
            relevance_score=0.7
        ))
    elif duration_seconds <= 900:  # 15 minutes or less
        mid_point = duration_seconds // 2
        segments.extend([
            VideoSegment(
                start_time="1:00",
                end_time=format_duration(mid_point),
                topic=f"{topic} - Part 1",
                relevance_score=0.7
            ),
            VideoSegment(
                start_time=format_duration(mid_point + 30),
                end_time=format_duration(duration_seconds - 30),
                topic=f"{topic} - Part 2",
                relevance_score=0.6
            )
        ])
    else:  # Longer videos
        third = duration_seconds // 3
        segments.extend([
            VideoSegment(
                start_time="2:00",
                end_time=format_duration(third),
                topic=f"{topic} - Introduction",
                relevance_score=0.7
            ),
            VideoSegment(
                start_time=format_duration(third + 60),
                end_time=format_duration(third * 2),
                topic=f"{topic} - Core Concepts",
                relevance_score=0.6
            ),
            VideoSegment(
                start_time=format_duration(third * 2 + 60),
                end_time=format_duration(duration_seconds - 60),
                topic=f"{topic} - Advanced Topics",
                relevance_score=0.5
            )
        ])
    
    return segments

def simple_keyword_score(video_title: str, video_description: str, keywords: List[str]) -> float:
    """Simple keyword matching scoring method."""
    if not keywords:
        return 0.3
    
    title_lower = video_title.lower()
    desc_lower = (video_description or "").lower()
    
    matches = 0
    for keyword in keywords:
        if keyword.lower() in title_lower:
            matches += 2
        elif keyword.lower() in desc_lower:
            matches += 1
    
    max_possible = len(keywords) * 2
    score = min(1.0, matches / max_possible) if max_possible > 0 else 0.3
    return max(0.2, score)

def calculate_relevance_score(video_details: Dict[str, Any], keywords: List[str]) -> float:
    """Calculate video relevance score."""
    score = 0.0
    
    # Keyword matching
    keyword_score = simple_keyword_score(
        video_details.get("title", ""),
        video_details.get("description", ""),
        keywords
    )
    score += keyword_score * 0.6
    
    # View count factor
    view_count = video_details.get("view_count", 0)
    if view_count > 1000:
        score += 0.2
    elif view_count > 100:
        score += 0.1
    
    # Duration appropriateness
    duration_seconds = parse_duration_to_seconds(video_details.get("duration", "PT0S"))
    if 60 <= duration_seconds <= 3600:
        score += 0.2
    elif duration_seconds > 0:
        score += 0.1
    
    return max(0.2, score)

def search_youtube(query: str, max_duration_minutes: Optional[int]) -> List[Dict]:
    """Search YouTube with a single optimized query."""
    try:
        search_params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 10,
            "type": "video",
            "relevanceLanguage": "en",
            "order": "relevance",
            "safeSearch": "moderate"
        }
        
        # Add duration filter if specified
        if max_duration_minutes:
            if max_duration_minutes <= 4:
                search_params["videoDuration"] = "short"
            elif max_duration_minutes <= 20:
                search_params["videoDuration"] = "medium"
        
        response = requests.get(YOUTUBE_SEARCH_URL, params=search_params, timeout=10)
        response.raise_for_status()
        
        search_data = response.json()
        items = search_data.get("items", [])
        
        # Add query info to each item
        for item in items:
            item["query_used"] = query
            
        return items
        
    except Exception as e:
        logger.warning(f"YouTube search failed: {e}")
        return []

def get_video_details(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch detailed video information."""
    try:
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY
        }
        
        response = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        video_details = {}
        
        for item in data.get("items", []):
            video_id = item["id"]
            snippet = item["snippet"]
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})
            
            video_details[video_id] = {
                "title": snippet["title"],
                "channel_title": snippet["channelTitle"],
                "published_at": snippet["publishedAt"],
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "duration": content_details.get("duration", "PT0M0S"),
                "view_count": int(statistics.get("viewCount", 0)),
                "description": snippet.get("description", "")
            }
            
        return video_details
        
    except Exception as e:
        logger.error(f"Error fetching video details: {e}")
        return {}

def get_best_youtube_video(youtube_topic: str, subject: Optional[str], level: str, max_duration_minutes: Optional[int]) -> Optional[YouTubeVideo]:
    """Get the best YouTube video for the given topic."""
    try:
        # Generate search query
        query = generate_search_query(youtube_topic, subject, level)
        logger.info(f"YouTube search query: {query}")
        
        # Search YouTube
        search_results = search_youtube(query, max_duration_minutes)
        
        if not search_results:
            logger.warning("No YouTube search results found")
            return None
        
        # Get video IDs
        video_ids = [item["id"]["videoId"] for item in search_results]
        
        # Get detailed video information
        video_details = get_video_details(video_ids)
        
        if not video_details:
            logger.warning("Could not retrieve video details")
            return None
        
        # Score and select best video
        keywords = extract_keywords_simple(youtube_topic, subject)
        scored_videos = []
        
        for video_id, details in video_details.items():
            relevance_score = calculate_relevance_score(details, keywords)
            scored_videos.append((video_id, details, relevance_score))
        
        # Sort by relevance score and select best
        scored_videos.sort(key=lambda x: x[2], reverse=True)
        best_video_id, best_details, best_score = scored_videos[0]
        
        logger.info(f"Selected video: {best_details['title']} (score: {best_score})")
        
        # Create segments
        duration_seconds = parse_duration_to_seconds(best_details["duration"])
        segments = create_default_segments(youtube_topic, duration_seconds)
        
        # Format response
        formatted_duration = format_duration(duration_seconds)
        
        return YouTubeVideo(
            title=best_details["title"],
            video_id=best_video_id,
            url=f"https://www.youtube.com/watch?v={best_video_id}",
            thumbnail=best_details["thumbnail"],
            duration=formatted_duration,
            channel_title=best_details["channel_title"],
            view_count=best_details["view_count"],
            published_at=best_details["published_at"],
            query_used=query,
            relevant_segments=segments,
            overall_relevance_score=round(best_score, 2)
        )
        
    except Exception as e:
        logger.error(f"Error getting YouTube video: {e}")
        return None

# --- Teaching Helper Functions --- #
def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))

def _block_valid(block: TeachingBlock) -> tuple[bool, Optional[str]]:
    if not isinstance(block.title, str) or not block.title.strip():
        return False, "missing or invalid title"
    if not isinstance(block.content, str) or not block.content.strip():
        return False, "missing or invalid content"
    words = _count_words(block.content)
    if words < 40:
        return False, f"content too short ({words} words)"
    if len(block.title) > 200:
        return False, "title too long"
    return True, None

def _sanitize_content(raw: str) -> str:
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", "\\n")
    s = s.replace('"', '\\"')
    return s

def _validate_and_sanitize_blocks(blocks: List[TeachingBlock]) -> tuple[bool, Optional[str], List[TeachingBlock]]:
    sanitized = []
    for i, b in enumerate(blocks):
        if not isinstance(b.title, str) or not isinstance(b.content, str):
            return False, f"block {i} has invalid types", blocks
        temp_block = TeachingBlock(title=b.title, content=b.content)
        ok, reason = _block_valid(temp_block)
        if not ok:
            return False, f"block {i} invalid: {reason}", blocks
        sanitized_content = _sanitize_content(b.content)
        sanitized.append(TeachingBlock(title=b.title, content=sanitized_content))
    return True, None, sanitized

def _call_model_and_get_parsed(input_messages, max_tokens=4000):
    return client.responses.parse(
        model="gpt-5-nano",
        input=input_messages,
        text_format=TeachingResponse,
        reasoning={"effort": "low"},
        instructions="Teach the topic in a casual, and conversational style that mimics how a tutor would explain things. Keep tone engaging throughout entire lesson, especially in the later blocks",
        max_output_tokens=max_tokens,
    )

def generate_teaching_content(req: CombinedRequest) -> List[TeachingBlock]:
    """Generate teaching content using the existing teaching module logic."""
    try:
        # Context info
        context_parts = []
        if req.subject:
            context_parts.append(f"Subject: {req.subject}")
        if req.level:
            context_parts.append(f"Level: {req.level}")
        if req.test_type:
            context_parts.append(f"Test: {req.test_type}")
        context_info = "\n".join(context_parts)

        # User prompt
        user_prompt = f"""{context_info}

Create a comprehensive lesson based on this study plan: {req.teaching_description}

Ensure every topic in the study plan is properly explained, and avoid veering from the study plan.
Output exactly 8-14 teaching blocks in valid JSON format with proper formatting including bullet points and bold text.
"""

        # Messages
        input_messages = [
            {"role": "system", "content": GPT_SYSTEM_PROMPT},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_1},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_2},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_3},
            {"role": "assistant", "content": ASSISTANT_EXAMPLE_JSON_4},
            {"role": "user", "content": user_prompt},
        ]

        # First attempt
        response = _call_model_and_get_parsed(input_messages)

        if getattr(response, "output_parsed", None) is None:
            if hasattr(response, "refusal") and response.refusal:
                raise HTTPException(status_code=400, detail=response.refusal)
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: If the previous response had any formatting or schema issues, return only the corrected single JSON object. Nothing else."
            }
            response = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail="Model did not return valid parsed output after retry.")

        lesson_blocks = response.output_parsed.lesson

        # Ensure 8–14 blocks
        if not (8 <= len(lesson_blocks) <= 14):
            retry_msg = {
                "role": "user",
                "content": "Fix JSON only: Must have 8-14 blocks. Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Lesson block count invalid ({len(lesson_blocks)}). Retry failed.")
            lesson_blocks = response_retry.output_parsed.lesson
            if not (8 <= len(lesson_blocks) <= 14):
                raise HTTPException(status_code=500, detail=f"Lesson block count invalid after retry ({len(lesson_blocks)}).")

        # Validate + sanitize
        valid, reason, sanitized_blocks = _validate_and_sanitize_blocks(lesson_blocks)
        if not valid:
            retry_msg = {
                "role": "user",
                "content": f"Fix JSON only: Last output failed validation ({reason}). Return corrected JSON only."
            }
            response_retry = _call_model_and_get_parsed(input_messages + [retry_msg])
            if getattr(response_retry, "output_parsed", None) is None:
                raise HTTPException(status_code=500, detail=f"Validation failed ({reason}) and retry failed.")
            lesson_blocks = response_retry.output_parsed.lesson
            valid2, reason2, sanitized_blocks2 = _validate_and_sanitize_blocks(lesson_blocks)
            if not valid2:
                raise HTTPException(status_code=500, detail=f"Validation failed after retry: {reason2}")
            sanitized_blocks = sanitized_blocks2

        return sanitized_blocks

    except Exception as e:
        logger.error(f"Error generating teaching content: {e}")
        raise e

# --- Main Combined Endpoint --- #
@router.post("/combined", response_model=CombinedResponse)
async def get_combined_content(req: CombinedRequest):
    """
    Generate both teaching content and YouTube video recommendation in parallel.
    Optimized for cases where YouTube topic is pre-specified.
    """
    try:
        logger.info(f"Processing combined request - Subject: {req.subject}, YouTube topic: {req.youtube_topic}")
        
        # Use ThreadPoolExecutor to run both operations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            teaching_future = executor.submit(generate_teaching_content, req)
            youtube_future = executor.submit(
                get_best_youtube_video, 
                req.youtube_topic, 
                req.subject, 
                req.level, 
                req.max_duration_minutes
            )
            
            # Wait for both to complete
            teaching_blocks = teaching_future.result()
            youtube_video = youtube_future.result()
        
        # Determine status
        if youtube_video:
            status = "success"
        else:
            status = "partial_success"  # Teaching worked but YouTube failed
            logger.warning("YouTube video search failed, returning teaching content only")
        
        return CombinedResponse(
            lesson=teaching_blocks,
            youtube_video=youtube_video,
            status=status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in combined endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Service temporarily unavailable: {str(e)}")

# --- Health Check Endpoint --- #
@router.get("/combined/health")
async def health_check():
    """Health check endpoint to verify API keys and service status."""
    try:
        # Test YouTube API
        test_params = {
            "part": "snippet",
            "q": "test tutorial",
            "key": YOUTUBE_API_KEY,
            "maxResults": 1,
            "type": "video"
        }
        response = requests.get(YOUTUBE_SEARCH_URL, params=test_params, timeout=5)
        youtube_status = "ok" if response.status_code == 200 else "error"
        
        return {
            "status": "healthy" if youtube_status == "ok" else "degraded",
            "youtube_api": youtube_status,
            "openai_api": "ok" if OPENAI_API_KEY else "missing",
            "features": {
                "parallel_processing": "enabled",
                "teaching_generation": "available",
                "youtube_search": "optimized_for_specified_topics",
                "combined_response": "available"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
