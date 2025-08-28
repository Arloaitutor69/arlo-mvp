# youtube.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import os
from openai import OpenAI
import logging

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Keys --- #
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Input schema --- #
class YouTubeRequest(BaseModel):
    teaching_material: str  # The actual teaching content/material
    subject: Optional[str] = None
    level: Optional[str] = "intermediate"  # beginner, intermediate, advanced
    max_duration_minutes: Optional[int] = 30  # Filter videos by max duration

# --- Output schema --- #
class VideoSegment(BaseModel):
    start_time: str  # Format: "MM:SS" or "HH:MM:SS"
    end_time: str
    topic: str
    relevance_score: float  # 0.0 to 1.0

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

def extract_key_topics(teaching_material: str, subject: Optional[str], level: str) -> List[str]:
    """
    Uses GPT-5-nano to extract 3-5 key topics from teaching material.
    """
    if not teaching_material or not teaching_material.strip():
        return [subject] if subject else ["General tutorial"]
    
    context_parts = [f"Teaching Material: {teaching_material}"]
    if subject:
        context_parts.append(f"Subject: {subject}")
    context_parts.append(f"Level: {level}")
    
    context_text = "\n".join(context_parts)
    
    prompt = f"""
Analyze this teaching material and extract the 3-5 most important topics that a student should understand.
Focus on concepts that would benefit from video explanation or demonstration.

{context_text}

Return ONLY a numbered list of key topics, one per line. Example:
1. Photosynthesis process
2. Chloroplast structure
3. Light and dark reactions
"""
    
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=150,
            reasoning={"effort": "low"}
        )
        
        topics_text = response.output_text.strip()
        topics = []
        for line in topics_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering/bullets and clean up
                topic = line.split('.', 1)[-1].split(')', 1)[-1].strip()
                if topic:
                    topics.append(topic)
        
        # Ensure we always have at least one topic
        if not topics:
            if subject:
                topics = [subject]
            else:
                # Extract first meaningful words from teaching material
                words = teaching_material.split()[:5]
                topics = [" ".join(words)]
        
        return topics[:5]  # Limit to 5 topics max
        
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        # Enhanced fallback logic
        fallback_topics = []
        if subject:
            fallback_topics.append(subject)
        
        # Try to extract meaningful terms from teaching material
        if teaching_material:
            words = teaching_material.split()
            meaningful_words = [word for word in words if len(word) > 3 and word.isalpha()]
            if meaningful_words:
                fallback_topics.extend(meaningful_words[:3])
        
        return fallback_topics if fallback_topics else ["tutorial"]

def generate_search_query(topics: List[str], subject: Optional[str], level: str) -> str:
    """
    Creates an optimized YouTube search query from key topics.
    """
    if not topics:
        topics = [subject] if subject else ["tutorial"]
    
    query_parts = []
    if subject:
        query_parts.append(subject)
    
    # Use top 2 topics, ensure they exist
    available_topics = [t for t in topics[:2] if t and t.strip()]
    query_parts.extend(available_topics)
    
    # Add level-specific terms
    if level == "beginner":
        query_parts.append("explained simply")
    elif level == "advanced":
        query_parts.append("detailed analysis")
    else:
        query_parts.append("tutorial")
    
    # Ensure we have at least something to search for
    if not query_parts:
        query_parts = ["educational tutorial"]
    
    query = " ".join(query_parts)
    return query[:100]  # YouTube has query length limits

def analyze_video_segments(video_title: str, video_description: str, duration: str, 
                         topics: List[str], teaching_material: str) -> List[VideoSegment]:
    """
    Uses GPT-5-nano to predict relevant time segments in the video.
    """
    if not topics:
        topics = ["main content"]
    
    # Limit description length to prevent token issues
    description = video_description[:500] if video_description else "No description available"
    
    prompt = f"""
You are analyzing a YouTube video to find segments relevant to specific teaching topics.

Video Title: {video_title}
Video Description: {description}
Video Duration: {duration}
Teaching Topics: {', '.join(topics[:3])}

Based on the video title and description, predict the most likely time segments where each topic might be covered.
Consider typical video structures (intro, main content sections, conclusion).

Return segments in this exact format:
Topic: [topic name]
Start: MM:SS
End: MM:SS
Relevance: 0.X
---

Limit to 3 most relevant segments. Use realistic time estimates.
"""

    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=300,
            reasoning={"effort": "medium"}
        )
        
        segments_text = response.output_text.strip()
        segments = []
        
        current_segment = {}
        for line in segments_text.split('\n'):
            line = line.strip()
            if line.startswith('Topic:'):
                if current_segment and len(current_segment) >= 4:
                    segments.append(VideoSegment(**current_segment))
                current_segment = {'topic': line.replace('Topic:', '').strip()}
            elif line.startswith('Start:'):
                current_segment['start_time'] = line.replace('Start:', '').strip()
            elif line.startswith('End:'):
                current_segment['end_time'] = line.replace('End:', '').strip()
            elif line.startswith('Relevance:'):
                try:
                    score = float(line.replace('Relevance:', '').strip())
                    current_segment['relevance_score'] = min(1.0, max(0.0, score))
                except:
                    current_segment['relevance_score'] = 0.7
            elif line == '---' and current_segment and len(current_segment) >= 4:
                segments.append(VideoSegment(**current_segment))
                current_segment = {}
        
        # Add final segment if exists and complete
        if current_segment and len(current_segment) >= 4:
            segments.append(VideoSegment(**current_segment))
        
        # Return segments or fallback
        if segments:
            return segments[:3]
        else:
            # Create fallback segments
            return [VideoSegment(
                start_time="00:30",
                end_time="05:00", 
                topic=topics[0] if topics else "Main content",
                relevance_score=0.7
            )]
        
    except Exception as e:
        logger.error(f"Error analyzing segments: {e}")
        # Enhanced fallback segments based on topics
        fallback_segments = []
        for i, topic in enumerate(topics[:2]):
            start_min = 1 + i * 3
            end_min = start_min + 4
            fallback_segments.append(VideoSegment(
                start_time=f"{start_min:02d}:00",
                end_time=f"{end_min:02d}:00",
                topic=topic,
                relevance_score=0.6
            ))
        
        return fallback_segments if fallback_segments else [VideoSegment(
            start_time="01:00",
            end_time="05:00",
            topic="Main content",
            relevance_score=0.5
        )]

def get_video_details(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch detailed video information including duration, view count, etc.
    """
    try:
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY
        }
        
        response = requests.get(YOUTUBE_VIDEOS_URL, params=params)
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

def parse_duration_to_seconds(duration: str) -> int:
    """
    Convert YouTube duration format (PT1H2M3S) to seconds.
    """
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds

def format_duration(seconds: int) -> str:
    """
    Convert seconds to MM:SS or HH:MM:SS format.
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def calculate_relevance_score(video_details: Dict[str, Any], topics: List[str], 
                            teaching_material: str) -> float:
    """
    Calculate overall relevance score for the video based on multiple factors.
    """
    if not topics:
        topics = ["tutorial"]  # Fallback to prevent division by zero
    
    score = 0.0
    
    # Title relevance
    title = video_details.get("title", "").lower()
    topic_matches = sum(1 for topic in topics if topic and topic.lower() in title)
    
    # Prevent division by zero
    if len(topics) > 0:
        score += (topic_matches / len(topics)) * 0.4
    
    # View count factor (normalized)
    view_count = video_details.get("view_count", 0)
    if view_count > 100000:
        score += 0.2
    elif view_count > 10000:
        score += 0.1
    
    # Duration appropriateness
    duration_seconds = parse_duration_to_seconds(video_details.get("duration", "PT0S"))
    if 300 <= duration_seconds <= 1800:  # 5-30 minutes is ideal
        score += 0.2
    elif duration_seconds <= 300:
        score += 0.1
    
    # Channel authority (simple heuristic)
    channel = video_details.get("channel_title", "").lower()
    if any(term in channel for term in ["university", "academy", "education", "khan", "crash course"]):
        score += 0.2
    
    # Base score to ensure minimum relevance
    score = max(score, 0.1)
    
    return min(1.0, score)

@router.post("/youtube", response_model=YouTubeVideo)
async def get_best_youtube_video(req: YouTubeRequest):
    """
    Find the most relevant YouTube video for given teaching material.
    """
    try:
        logger.info(f"Processing request for subject: {req.subject}")
        
        # Step 1: Extract key topics from teaching material
        topics = extract_key_topics(req.teaching_material, req.subject, req.level)
        logger.info(f"Extracted topics: {topics}")
        
        # Step 2: Generate optimized search query
        search_query = generate_search_query(topics, req.subject, req.level)
        logger.info(f"Search query: {search_query}")
        
        # Step 3: Search YouTube
        search_params = {
            "part": "snippet",
            "q": search_query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 5,
            "type": "video",
            "safeSearch": "strict",
            "relevanceLanguage": "en",
            "order": "relevance"
        }
        
        # Add duration filter if specified
        if req.max_duration_minutes:
            if req.max_duration_minutes <= 4:
                search_params["videoDuration"] = "short"
            elif req.max_duration_minutes <= 20:
                search_params["videoDuration"] = "medium"
        
        response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
        response.raise_for_status()
        
        search_data = response.json()
        items = search_data.get("items", [])
        
        if not items:
            raise HTTPException(status_code=404, detail="No relevant YouTube videos found.")
        
        # Step 4: Get detailed video information
        video_ids = [item["id"]["videoId"] for item in items]
        video_details = get_video_details(video_ids)
        
        # Step 5: Score and rank videos
        scored_videos = []
        for item in items:
            video_id = item["id"]["videoId"]
            if video_id in video_details:
                details = video_details[video_id]
                relevance_score = calculate_relevance_score(details, topics, req.teaching_material)
                scored_videos.append((video_id, details, relevance_score))
        
        if not scored_videos:
            raise HTTPException(status_code=404, detail="No valid video details found.")
        
        # Sort by relevance score
        scored_videos.sort(key=lambda x: x[2], reverse=True)
        best_video_id, best_details, best_score = scored_videos[0]
        
        # Step 6: Analyze video segments
        segments = analyze_video_segments(
            best_details["title"],
            best_details.get("description", ""),
            best_details["duration"],
            topics,
            req.teaching_material
        )
        
        # Step 7: Format response
        duration_seconds = parse_duration_to_seconds(best_details["duration"])
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
            query_used=search_query,
            relevant_segments=segments,
            overall_relevance_score=round(best_score, 2)
        )
        
    except HTTPException:
        raise
    except requests.RequestException as e:
        logger.error(f"YouTube API request failed: {e}")
        raise HTTPException(status_code=503, detail="YouTube API temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Health check endpoint
@router.get("/youtube/health")
async def health_check():
    """
    Health check endpoint to verify API keys and service status.
    """
    try:
        # Test YouTube API
        test_params = {
            "part": "snippet",
            "q": "test",
            "key": YOUTUBE_API_KEY,
            "maxResults": 1,
            "type": "video"
        }
        response = requests.get(YOUTUBE_SEARCH_URL, params=test_params)
        youtube_status = "ok" if response.status_code == 200 else "error"
        
        # Test OpenAI API
        try:
            test_response = client.responses.create(
                model="gpt-5-nano",
                input=[{"role": "user", "content": "test"}],
                max_output_tokens=5,
                reasoning={"effort": "low"}
            )
            openai_status = "ok"
        except:
            openai_status = "error"
        
        return {
            "status": "healthy" if youtube_status == "ok" and openai_status == "ok" else "unhealthy",
            "youtube_api": youtube_status,
            "openai_api": openai_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
