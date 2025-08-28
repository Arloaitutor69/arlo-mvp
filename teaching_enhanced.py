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
import math

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Environment Variables --- #
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Trusted Educational Channels --- #
EDUCATIONAL_CHANNELS = {
    "Khan Academy": {"boost": 0.3, "keywords": ["khan academy", "khanacademy"]},
    "Crash Course": {"boost": 0.25, "keywords": ["crash course", "crashcourse"]},
    "Organic Chemistry Tutor": {"boost": 0.25, "keywords": ["organic chemistry tutor", "organicchemistrytutor"]},
}

# --- Input Schema --- #
class CombinedRequest(BaseModel):
    teaching_description: str  # Main teaching content description - will be parsed for YouTube search
    subject: Optional[str] = None
    level: Optional[str] = "intermediate"
    test_type: Optional[str] = None

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
    quality_score: float
    final_score: float

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
def extract_youtube_search_from_teaching_content(teaching_description: str) -> Optional[str]:
    """Extract YouTube search query from teaching content after 'Watch Youtube video about'."""
    pattern = r"Watch Youtube video about\s+(.+?)(?:\n|$|\.)"
    match = re.search(pattern, teaching_description, re.IGNORECASE)
    if match:
        youtube_query = match.group(1).strip()
        # Clean up common sentence endings
        youtube_query = re.sub(r'[.!?]+$', '', youtube_query)
        logger.info(f"Extracted YouTube search: '{youtube_query}'")
        return youtube_query
    
    logger.info("No 'Watch Youtube video about' found in teaching content")
    return None

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

def seconds_to_time_format(seconds: int) -> str:
    """Convert seconds to M:SS or MM:SS format for timestamps."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}:{secs:02d}"

def time_format_to_seconds(time_str: str) -> int:
    """Convert M:SS or MM:SS format to seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    except:
        return 0
    return 0

def parse_description_timestamps(description: str, search_query: str) -> List[VideoSegment]:
    """
    Parse video description for manual timestamps that match the search query.
    Returns relevant segments based on keyword matching.
    """
    segments = []
    if not description or not search_query:
        return segments
    
    # Common timestamp patterns
    timestamp_patterns = [
        r'(\d{1,2}:\d{2})\s*[-–—]\s*([^\n\r]+)',  # 1:30 - Topic description
        r'(\d{1,2}:\d{2})\s+([^\n\r]+)',          # 1:30 Topic description
        r'(\d{1,2}:\d{2}:\d{2})\s*[-–—]\s*([^\n\r]+)',  # 1:30:00 - Topic description
        r'(\d{1,2}:\d{2}:\d{2})\s+([^\n\r]+)',          # 1:30:00 Topic description
    ]
    
    search_terms = [term.lower().strip() for term in search_query.split()]
    
    for pattern in timestamp_patterns:
        matches = re.finditer(pattern, description, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            timestamp = match.group(1)
            topic_text = match.group(2).strip()
            
            # Skip if topic text is too short or looks like noise
            if len(topic_text) < 5 or topic_text.lower() in ['intro', 'introduction', 'outro', 'end']:
                continue
            
            # Calculate relevance score based on keyword matching
            relevance_score = 0.0
            topic_lower = topic_text.lower()
            
            # Exact query match gets highest score
            if search_query.lower() in topic_lower:
                relevance_score = 0.95
            else:
                # Check individual terms
                for term in search_terms:
                    if len(term) > 2 and term in topic_lower:  # Skip very short terms
                        relevance_score += 0.3
                        
                # Bonus for educational keywords
                educational_keywords = ['explains', 'tutorial', 'how to', 'lesson', 'example', 'definition']
                for keyword in educational_keywords:
                    if keyword in topic_lower:
                        relevance_score += 0.1
            
            # Only include segments with reasonable relevance
            if relevance_score >= 0.3:
                start_seconds = time_format_to_seconds(timestamp)
                
                # Estimate end time (1-2 minutes from start, or until next timestamp)
                end_seconds = start_seconds + 90  # Default 1.5 minutes
                
                # Try to find next timestamp to set better end time
                next_timestamps = re.findall(r'\d{1,2}:\d{2}(?::\d{2})?', description[match.end():])
                if next_timestamps:
                    next_start = time_format_to_seconds(next_timestamps[0])
                    if next_start > start_seconds:
                        end_seconds = min(start_seconds + 120, next_start)  # Max 2 minutes or until next
                
                segments.append(VideoSegment(
                    start_time=seconds_to_time_format(start_seconds),
                    end_time=seconds_to_time_format(end_seconds),
                    topic=topic_text[:100],  # Limit topic length
                    relevance_score=round(relevance_score, 2)
                ))
    
    # Sort by relevance score and return top segments
    segments.sort(key=lambda x: x.relevance_score, reverse=True)
    return segments[:3]  # Return top 3 most relevant segments

def create_smart_segments(youtube_query: str, duration_seconds: int, description: str = "") -> List[VideoSegment]:
    """
    Create intelligent video segments using description timestamps or fallback to whole video.
    """
    # First, try to parse description for relevant timestamps
    if description and youtube_query:
        parsed_segments = parse_description_timestamps(description, youtube_query)
        if parsed_segments:
            logger.info(f"Found {len(parsed_segments)} relevant timestamp segments in description")
            return parsed_segments
    
    # Fallback: For short videos (under 5 minutes), show entire video as one segment
    if duration_seconds <= 300:  # 5 minutes
        topic = youtube_query if youtube_query else "Complete video content"
        return [VideoSegment(
            start_time="0:00",
            end_time=format_duration(duration_seconds),
            topic=f"{topic} (Full video)",
            relevance_score=0.8
        )]
    
    # For longer videos without timestamps, create two balanced segments
    mid_point = duration_seconds // 2
    topic = youtube_query if youtube_query else "Video content"
    
    return [
        VideoSegment(
            start_time="0:30",
            end_time=seconds_to_time_format(mid_point),
            topic=f"{topic} - Part 1",
            relevance_score=0.7
        ),
        VideoSegment(
            start_time=seconds_to_time_format(mid_point + 15),
            end_time=seconds_to_time_format(duration_seconds - 15),
            topic=f"{topic} - Part 2",
            relevance_score=0.7
        )
    ]

def is_educational_channel(channel_title: str) -> tuple[bool, float]:
    """Check if channel is a trusted educational channel and return boost score."""
    channel_lower = channel_title.lower()
    
    for edu_channel, config in EDUCATIONAL_CHANNELS.items():
        for keyword in config["keywords"]:
            if keyword in channel_lower:
                return True, config["boost"]
    
    return False, 0.0

def calculate_quality_score(video_details: Dict[str, Any], channel_info: Dict[str, Any] = None) -> float:
    """Calculate video quality score based on multiple factors."""
    quality_score = 0.0
    
    # View count (logarithmic scaling)
    view_count = video_details.get("view_count", 0)
    if view_count > 0:
        log_views = math.log10(max(view_count, 1))
        # Scale: 1K views = 0.1, 10K = 0.2, 100K = 0.3, 1M = 0.4, 10M = 0.5
        view_score = min(0.5, log_views / 10)
        quality_score += view_score
    
    # Educational channel boost
    is_edu, edu_boost = is_educational_channel(video_details.get("channel_title", ""))
    if is_edu:
        quality_score += edu_boost
    
    # Engagement rate (if available)
    like_count = video_details.get("like_count", 0)
    if view_count > 0 and like_count > 0:
        engagement_rate = like_count / view_count
        # Boost for high engagement (typically 1-5% is good)
        if engagement_rate > 0.02:  # 2%+
            quality_score += 0.15
        elif engagement_rate > 0.01:  # 1%+
            quality_score += 0.1
    
    # Duration appropriateness (prefer 2-3 minutes, max 5 minutes)
    duration_seconds = parse_duration_to_seconds(video_details.get("duration", "PT0S"))
    if 120 <= duration_seconds <= 180:  # 2-3 minutes (ideal)
        quality_score += 0.2
    elif 60 <= duration_seconds <= 300:  # 1-5 minutes (acceptable)
        quality_score += 0.1
    elif duration_seconds > 300:  # Over 5 minutes (penalize)
        quality_score -= 0.1
    
    # Content quality indicators
    description = video_details.get("description", "")
    if len(description) > 200:  # Detailed description
        quality_score += 0.05
    
    # Bonus for descriptions with timestamps (indicates structured content)
    if re.search(r'\d{1,2}:\d{2}', description):
        quality_score += 0.1
    
    # Channel authority (subscriber count if available)
    if channel_info and "subscriber_count" in channel_info:
        subscriber_count = channel_info["subscriber_count"]
        if subscriber_count > 1000000:  # 1M+ subscribers
            quality_score += 0.15
        elif subscriber_count > 100000:  # 100K+ subscribers
            quality_score += 0.1
        elif subscriber_count > 10000:  # 10K+ subscribers
            quality_score += 0.05
    
    return min(1.0, quality_score)  # Cap at 1.0

def calculate_relevance_score(video_details: Dict[str, Any], search_query: str) -> float:
    """Calculate video relevance score based on keyword matching."""
    if not search_query:
        return 0.3
    
    title = video_details.get("title", "").lower()
    description = video_details.get("description", "").lower()
    search_terms = search_query.lower().split()
    
    relevance_score = 0.0
    
    # Title matching (higher weight)
    for term in search_terms:
        if term in title:
            relevance_score += 0.15
    
    # Description matching (lower weight)
    for term in search_terms:
        if term in description:
            relevance_score += 0.05
    
    # Exact phrase matching (bonus)
    if search_query.lower() in title:
        relevance_score += 0.2
    elif search_query.lower() in description:
        relevance_score += 0.1
    
    return min(1.0, relevance_score)

def get_channel_info(channel_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch channel information for quality assessment."""
    try:
        if not channel_ids:
            return {}
        
        params = {
            "part": "statistics",
            "id": ",".join(channel_ids),
            "key": YOUTUBE_API_KEY
        }
        
        response = requests.get(YOUTUBE_CHANNELS_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        channel_info = {}
        
        for item in data.get("items", []):
            channel_id = item["id"]
            stats = item.get("statistics", {})
            
            channel_info[channel_id] = {
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "video_count": int(stats.get("videoCount", 0)),
                "view_count": int(stats.get("viewCount", 0))
            }
        
        return channel_info
        
    except Exception as e:
        logger.warning(f"Error fetching channel info: {e}")
        return {}

def search_youtube(query: str) -> List[Dict]:
    """Search YouTube with quality filters."""
    try:
        search_params = {
            "part": "snippet",
            "q": query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 15,  # Get more to filter better
            "type": "video",
            "relevanceLanguage": "en",
            "regionCode": "US",  # Ensure US region for English content
            "order": "relevance",
            "safeSearch": "moderate",
            "videoDuration": "short"  # Max 4 minutes
        }
        
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
                "channel_id": snippet["channelId"],
                "published_at": snippet["publishedAt"],
                "thumbnail": snippet["thumbnails"]["high"]["url"],
                "duration": content_details.get("duration", "PT0M0S"),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
                "description": snippet.get("description", "")
            }
            
        return video_details
        
    except Exception as e:
        logger.error(f"Error fetching video details: {e}")
        return {}

def get_best_youtube_video(youtube_query: str, subject: Optional[str] = None) -> Optional[YouTubeVideo]:
    """Get the best YouTube video for the given query with smart segment analysis."""
    try:
        logger.info(f"YouTube search query: '{youtube_query}'")
        
        # Search YouTube
        search_results = search_youtube(youtube_query)
        
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
        
        # Get channel information
        channel_ids = list(set(details.get("channel_id") for details in video_details.values()))
        channel_info = get_channel_info(channel_ids)
        
        # Filter by duration (max 5 minutes = 300 seconds)
        filtered_videos = {}
        for video_id, details in video_details.items():
            duration_seconds = parse_duration_to_seconds(details["duration"])
            if duration_seconds <= 300:  # 5 minutes max
                filtered_videos[video_id] = details
        
        if not filtered_videos:
            logger.warning("No videos found under 5 minutes")
            return None
        
        # Quality threshold filter (minimum 1K views unless educational channel)
        quality_filtered_videos = {}
        for video_id, details in filtered_videos.items():
            view_count = details.get("view_count", 0)
            is_edu, _ = is_educational_channel(details.get("channel_title", ""))
            
            if is_edu or view_count >= 1000:  # Educational channels bypass view threshold
                quality_filtered_videos[video_id] = details
        
        if not quality_filtered_videos:
            logger.warning("No videos passed quality threshold")
            # Fallback to original filtered set
            quality_filtered_videos = filtered_videos
        
        # Score videos with composite approach
        scored_videos = []
        
        for video_id, details in quality_filtered_videos.items():
            relevance_score = calculate_relevance_score(details, youtube_query)
            
            channel_data = channel_info.get(details.get("channel_id"), {})
            quality_score = calculate_quality_score(details, channel_data)
            
            # Composite final score: 40% relevance + 40% quality + 20% freshness
            freshness_score = 0.1  # Default freshness (could implement date-based scoring)
            
            final_score = (relevance_score * 0.4) + (quality_score * 0.4) + (freshness_score * 0.2)
            
            scored_videos.append((video_id, details, relevance_score, quality_score, final_score))
        
        # Sort by final score and select best
        scored_videos.sort(key=lambda x: x[4], reverse=True)
        best_video_id, best_details, relevance_score, quality_score, final_score = scored_videos[0]
        
        logger.info(f"Selected video: {best_details['title']} (final score: {final_score:.2f}, relevance: {relevance_score:.2f}, quality: {quality_score:.2f})")
        
        # Create smart segments using description analysis
        duration_seconds = parse_duration_to_seconds(best_details["duration"])
        segments = create_smart_segments(youtube_query, duration_seconds, best_details.get("description", ""))
        
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
            query_used=youtube_query,
            relevant_segments=segments,
            overall_relevance_score=round(relevance_score, 2),
            quality_score=round(quality_score, 2),
            final_score=round(final_score, 2)
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
    YouTube search is extracted from teaching content after 'Watch Youtube video about'.
    """
    try:
        logger.info(f"Processing combined request - Subject: {req.subject}")
        
        # Extract YouTube search query from teaching content
        youtube_query = extract_youtube_search_from_teaching_content(req.teaching_description)
        
        # Use ThreadPoolExecutor to run both operations in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit teaching task
            teaching_future = executor.submit(generate_teaching_content, req)
            
            # Submit YouTube task only if query found
            youtube_future = None
            if youtube_query:
                youtube_future = executor.submit(get_best_youtube_video, youtube_query, req.subject)
            
            # Wait for teaching to complete
            teaching_blocks = teaching_future.result()
            
            # Wait for YouTube if it was submitted
            youtube_video = None
            if youtube_future:
                youtube_video = youtube_future.result()
        
        # Determine status
        if youtube_query and youtube_video:
            status = "success"
        elif youtube_query and not youtube_video:
            status = "partial_success"  # Teaching worked but YouTube failed
            logger.warning("YouTube video search failed, returning teaching content only")
        else:
            status = "success_no_youtube"  # No YouTube query found in teaching content
            logger.info("No YouTube query found in teaching content")
        
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
            "type": "video",
            "videoDuration": "short"
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
                "youtube_search": "quality_and_relevance_balanced",
                "combined_response": "available",
                "max_video_length": "5_minutes",
                "preferred_length": "2-3_minutes",
                "educational_channels": list(EDUCATIONAL_CHANNELS.keys()),
                "quality_filtering": "enabled",
                "smart_segments": "description_timestamps_and_fallback"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
