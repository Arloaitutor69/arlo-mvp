# youtube.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import requests
import os
from openai import OpenAI
import logging
import re

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

def extract_keywords_simple(teaching_material: str, subject: Optional[str]) -> List[str]:
    """
    Simple keyword extraction using regex and basic NLP techniques.
    Much more reliable than GPT calls.
    """
    keywords = []
    
    # Add subject if provided
    if subject:
        keywords.append(subject)
    
    if teaching_material:
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'this', 'that', 'these', 'those'}
        
        # Extract words that are likely important (3+ chars, not common words)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', teaching_material.lower())
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Count frequency and take most common
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords.extend([word for word, freq in sorted_words[:5]])
    
    # Fallback if no keywords found
    if not keywords:
        keywords = ["tutorial", "lesson"]
    
    return keywords[:5]

def extract_key_topics_with_fallback(teaching_material: str, subject: Optional[str], level: str) -> List[str]:
    """
    Try GPT extraction first, fall back to simple keyword extraction.
    """
    # Try simple keyword extraction first (more reliable)
    simple_keywords = extract_keywords_simple(teaching_material, subject)
    
    # Only try GPT if we have substantial content
    if len(teaching_material.strip()) > 50:
        try:
            context_parts = [f"Teaching Material: {teaching_material[:300]}"]  # Limit length
            if subject:
                context_parts.append(f"Subject: {subject}")
            context_parts.append(f"Level: {level}")
            
            context_text = "\n".join(context_parts)
            
            prompt = f"""
Extract 3-5 key topics from this teaching material. Be concise.

{context_text}

Return ONLY a numbered list:
1. Topic one
2. Topic two
3. Topic three
"""
            
            response = client.responses.create(
                model="gpt-5-nano",
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=100,
                reasoning={"effort": "minimal"}
            )
            
            topics_text = response.output_text.strip()
            topics = []
            for line in topics_text.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    topic = re.sub(r'^\d+\.?\s*', '', line).strip()
                    if topic:
                        topics.append(topic)
            
            # If GPT worked, use it; otherwise use simple keywords
            if topics:
                return topics[:5]
            else:
                return simple_keywords
                
        except Exception as e:
            logger.warning(f"GPT topic extraction failed, using simple keywords: {e}")
            return simple_keywords
    else:
        return simple_keywords

def generate_search_queries(topics: List[str], subject: Optional[str], level: str) -> List[str]:
    """
    Generate multiple search queries with progressive relaxation.
    """
    queries = []
    
    # Query 1: Specific with subject and main topics
    if subject and topics:
        query1_parts = [subject] + topics[:2]
        if level == "beginner":
            query1_parts.append("basics")
        elif level == "advanced":
            query1_parts.append("advanced")
        queries.append(" ".join(query1_parts)[:80])
    
    # Query 2: Just subject and main topic
    if subject and topics:
        queries.append(f"{subject} {topics[0]} tutorial"[:80])
    
    # Query 3: Main topics only
    if topics:
        queries.append(f"{topics[0]} explained"[:80])
    
    # Query 4: Very general fallback
    if subject:
        queries.append(f"{subject} tutorial"[:80])
    else:
        queries.append("educational tutorial")
    
    # Remove duplicates while preserving order
    unique_queries = []
    for q in queries:
        if q not in unique_queries:
            unique_queries.append(q)
    
    return unique_queries

def create_default_segments(topics: List[str], duration_seconds: int) -> List[VideoSegment]:
    """
    Create reasonable default segments without AI analysis.
    """
    segments = []
    
    if not topics:
        topics = ["Main content"]
    
    # Create segments based on video duration
    if duration_seconds <= 300:  # 5 minutes or less
        segments.append(VideoSegment(
            start_time="0:30",
            end_time=format_duration(min(duration_seconds - 10, 240)),
            topic=topics[0],
            relevance_score=0.7
        ))
    elif duration_seconds <= 900:  # 15 minutes or less
        # Two segments
        mid_point = duration_seconds // 2
        segments.extend([
            VideoSegment(
                start_time="1:00",
                end_time=format_duration(mid_point),
                topic=topics[0],
                relevance_score=0.7
            ),
            VideoSegment(
                start_time=format_duration(mid_point + 30),
                end_time=format_duration(duration_seconds - 30),
                topic=topics[1] if len(topics) > 1 else topics[0],
                relevance_score=0.6
            )
        ])
    else:  # Longer videos
        # Three segments
        third = duration_seconds // 3
        segments.extend([
            VideoSegment(
                start_time="2:00",
                end_time=format_duration(third),
                topic=topics[0],
                relevance_score=0.7
            ),
            VideoSegment(
                start_time=format_duration(third + 60),
                end_time=format_duration(third * 2),
                topic=topics[1] if len(topics) > 1 else topics[0],
                relevance_score=0.6
            ),
            VideoSegment(
                start_time=format_duration(third * 2 + 60),
                end_time=format_duration(duration_seconds - 60),
                topic=topics[2] if len(topics) > 2 else topics[0],
                relevance_score=0.5
            )
        ])
    
    return segments

def analyze_video_segments_optional(video_title: str, video_description: str, duration: str, 
                                  topics: List[str], duration_seconds: int) -> List[VideoSegment]:
    """
    Try AI analysis, fall back to default segments if it fails.
    """
    # Always have default segments ready
    default_segments = create_default_segments(topics, duration_seconds)
    
    # Try AI analysis only for reasonable-sized content
    if len(video_title) > 10 and len(topics) > 0:
        try:
            description = video_description[:300] if video_description else "No description"
            
            prompt = f"""
Predict time segments for this video based on title and description.

Title: {video_title}
Description: {description}
Duration: {duration}
Topics: {', '.join(topics[:2])}

Format (exactly):
Topic: [name]
Start: MM:SS
End: MM:SS
Relevance: 0.X
---

Max 2 segments, realistic times.
"""

            response = client.responses.create(
                model="gpt-5-nano",
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=200,
                reasoning={"effort": "minimal"}
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
                        current_segment['relevance_score'] = 0.6
                elif line == '---' and current_segment and len(current_segment) >= 4:
                    segments.append(VideoSegment(**current_segment))
                    current_segment = {}
            
            # Add final segment if complete
            if current_segment and len(current_segment) >= 4:
                segments.append(VideoSegment(**current_segment))
            
            # Return AI segments if we got any, otherwise default
            return segments if segments else default_segments
            
        except Exception as e:
            logger.warning(f"AI segment analysis failed, using defaults: {e}")
            return default_segments
    else:
        return default_segments

def simple_keyword_score(video_title: str, video_description: str, keywords: List[str]) -> float:
    """
    Simple keyword matching backup scoring method.
    """
    if not keywords:
        return 0.3  # Default score
    
    title_lower = video_title.lower()
    desc_lower = (video_description or "").lower()
    
    matches = 0
    for keyword in keywords:
        if keyword.lower() in title_lower:
            matches += 2  # Title matches are worth more
        elif keyword.lower() in desc_lower:
            matches += 1
    
    # Normalize score
    max_possible = len(keywords) * 2
    score = min(1.0, matches / max_possible) if max_possible > 0 else 0.3
    
    # Ensure minimum score
    return max(0.2, score)

def calculate_relevance_score_simplified(video_details: Dict[str, Any], keywords: List[str]) -> float:
    """
    Simplified relevance scoring with lower thresholds.
    """
    score = 0.0
    
    # Keyword matching (primary factor)
    keyword_score = simple_keyword_score(
        video_details.get("title", ""),
        video_details.get("description", ""),
        keywords
    )
    score += keyword_score * 0.6
    
    # View count factor (relaxed)
    view_count = video_details.get("view_count", 0)
    if view_count > 1000:  # Much lower threshold
        score += 0.2
    elif view_count > 100:
        score += 0.1
    
    # Duration appropriateness (relaxed)
    duration_seconds = parse_duration_to_seconds(video_details.get("duration", "PT0S"))
    if 60 <= duration_seconds <= 3600:  # 1 minute to 1 hour
        score += 0.2
    elif duration_seconds > 0:  # Any duration is better than none
        score += 0.1
    
    # Ensure minimum viable score
    return max(0.2, score)

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

def parse_duration_to_seconds(duration: str) -> int:
    """
    Convert YouTube duration format (PT1H2M3S) to seconds.
    """
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

def search_youtube_progressive(queries: List[str], max_duration_minutes: Optional[int]) -> List[Dict]:
    """
    Try multiple search queries with progressive relaxation of criteria.
    """
    all_results = []
    
    for i, query in enumerate(queries):
        try:
            logger.info(f"Trying search query {i+1}: {query}")
            
            # Base search parameters
            search_params = {
                "part": "snippet",
                "q": query,
                "key": YOUTUBE_API_KEY,
                "maxResults": 10 if i == 0 else 5,  # More results for first query
                "type": "video",
                "relevanceLanguage": "en",
                "order": "relevance"
            }
            
            # Add duration filter only for first two attempts
            if i < 2 and max_duration_minutes:
                if max_duration_minutes <= 4:
                    search_params["videoDuration"] = "short"
                elif max_duration_minutes <= 20:
                    search_params["videoDuration"] = "medium"
            
            # Remove safeSearch strict after first attempt
            if i == 0:
                search_params["safeSearch"] = "moderate"
            
            response = requests.get(YOUTUBE_SEARCH_URL, params=search_params, timeout=10)
            response.raise_for_status()
            
            search_data = response.json()
            items = search_data.get("items", [])
            
            if items:
                # Add query info to each item
                for item in items:
                    item["query_used"] = query
                all_results.extend(items)
                logger.info(f"Found {len(items)} videos with query: {query}")
                
                # If we got good results from early queries, we can stop
                if len(all_results) >= 5 and i < 2:
                    break
            else:
                logger.warning(f"No results for query: {query}")
                
        except Exception as e:
            logger.warning(f"Search failed for query '{query}': {e}")
            continue
    
    return all_results

@router.post("/youtube", response_model=YouTubeVideo)
async def get_best_youtube_video(req: YouTubeRequest):
    """
    Find the most relevant YouTube video for given teaching material.
    Progressive relaxation approach with better fallbacks.
    """
    try:
        logger.info(f"Processing request for subject: {req.subject}")
        
        # Step 1: Extract topics with fallback to keywords
        topics = extract_key_topics_with_fallback(req.teaching_material, req.subject, req.level)
        logger.info(f"Extracted topics/keywords: {topics}")
        
        # Step 2: Generate multiple search queries
        search_queries = generate_search_queries(topics, req.subject, req.level)
        logger.info(f"Generated queries: {search_queries}")
        
        # Step 3: Progressive search with relaxed criteria
        all_items = search_youtube_progressive(search_queries, req.max_duration_minutes)
        
        if not all_items:
            raise HTTPException(status_code=404, detail="No YouTube videos found with any search criteria.")
        
        # Step 4: Get detailed video information
        video_ids = []
        query_map = {}
        for item in all_items:
            video_id = item["id"]["videoId"]
            if video_id not in video_ids:  # Avoid duplicates
                video_ids.append(video_id)
                query_map[video_id] = item.get("query_used", search_queries[0])
        
        video_details = get_video_details(video_ids)
        
        if not video_details:
            raise HTTPException(status_code=404, detail="Could not retrieve video details.")
        
        # Step 5: Score videos with simplified, lenient scoring
        scored_videos = []
        for video_id, details in video_details.items():
            relevance_score = calculate_relevance_score_simplified(details, topics)
            scored_videos.append((video_id, details, relevance_score, query_map.get(video_id, search_queries[0])))
        
        # Sort by relevance score
        scored_videos.sort(key=lambda x: x[2], reverse=True)
        best_video_id, best_details, best_score, used_query = scored_videos[0]
        
        logger.info(f"Selected video: {best_details['title']} (score: {best_score})")
        
        # Step 6: Create segments (AI with fallback to default)
        duration_seconds = parse_duration_to_seconds(best_details["duration"])
        segments = analyze_video_segments_optional(
            best_details["title"],
            best_details.get("description", ""),
            best_details["duration"],
            topics,
            duration_seconds
        )
        
        # Step 7: Format response
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
            query_used=used_query,
            relevant_segments=segments,
            overall_relevance_score=round(best_score, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Service temporarily unavailable")

# Health check endpoint
@router.get("/youtube/health")
async def health_check():
    """
    Health check endpoint to verify API keys and service status.
    """
    try:
        # Test YouTube API with simple query
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
            "features": {
                "ai_topic_extraction": "available",
                "ai_segment_analysis": "available_with_fallback",
                "progressive_search": "enabled"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
