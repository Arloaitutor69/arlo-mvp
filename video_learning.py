# youtube.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import requests
import os
from openai import OpenAI

router = APIRouter()

# --- Keys --- #
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "AIzaSyAd8kcbap76McZS6e5ZlBTXPz-tZU9nAQs")
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Input schema --- #
class YouTubeRequest(BaseModel):
    description: str
    subject: Optional[str] = None
    level: Optional[str] = None


# --- Output schema --- #
class YouTubeVideo(BaseModel):
    title: str
    video_id: str
    url: str
    thumbnail: str
    query_used: str


def _generate_search_query(req: YouTubeRequest) -> str:
    """
    Uses gpt-5-nano to generate a focused YouTube search query.
    """
    context = []
    if req.subject:
        context.append(f"Subject: {req.subject}")
    if req.level:
        context.append(f"Level: {req.level}")
    context.append(f"Description: {req.description}")
    context_text = "\n".join(context)

    prompt = f"""
You are helping find the best YouTube video for teaching. 
From this context, create ONE very short but specific YouTube search query (5-12 words max).
The query should sound like something a student would type to find the clearest educational video.

Context:
{context_text}

Output ONLY the query text, nothing else.
"""

    response = client.responses.create(
        model="gpt-5-nano",
        input=[{"role": "user", "content": prompt}],
        max_output_tokens=50,
        reasoning={"effort": "low"}
    )

    query = response.output_text.strip()
    return query


@router.post("/youtube", response_model=YouTubeVideo)
def get_best_youtube_video(req: YouTubeRequest):
    try:
        # Step 1: Generate optimized query
        search_query = _generate_search_query(req)

        # Step 2: Call YouTube API
        params = {
            "part": "snippet",
            "q": search_query,
            "key": YOUTUBE_API_KEY,
            "maxResults": 3,   # fetch a few, we can pick the best
            "type": "video",
            "safeSearch": "strict",
            "relevanceLanguage": "en"
        }

        response = requests.get(YOUTUBE_SEARCH_URL, params=params)

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"YouTube API error: {response.text}")

        data = response.json()
        items = data.get("items", [])
        if not items:
            raise HTTPException(status_code=404, detail="No relevant YouTube video found.")

        # Step 3: Pick the first result for now
        video = items[0]
        video_id = video["id"]["videoId"]
        snippet = video["snippet"]

        return YouTubeVideo(
            title=snippet["title"],
            video_id=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
            thumbnail=snippet["thumbnails"]["high"]["url"],
            query_used=search_query
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching YouTube video: {str(e)}")
