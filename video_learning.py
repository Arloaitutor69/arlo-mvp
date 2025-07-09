from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import requests
import openai
import json
import logging

router = APIRouter()

# Load API keys from environment
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Request and Response Models ===

class VideoLearningRequest(BaseModel):
    topic: str

class VideoLearningResponse(BaseModel):
    video_id: str
    title: str
    start: int
    end: int
    segment_title: Optional[str] = None

# === Helper Functions ===

def search_youtube_video(topic: str) -> dict:
    """Search YouTube for the most relevant video on a given topic."""
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "maxResults": 3,
        "key": YOUTUBE_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if "items" not in data or not data["items"]:
            raise HTTPException(status_code=404, detail="No YouTube videos found.")
        return data["items"][0]
    except requests.RequestException as e:
        logging.error(f"YouTube search failed: {e}")
        raise HTTPException(status_code=500, detail="YouTube search error.")

def get_video_description(video_id: str) -> str:
    """Fetch the video description using YouTube's video endpoint."""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails",
        "id": video_id,
        "key": YOUTUBE_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        return data["items"][0]["snippet"]["description"]
    except Exception as e:
        logging.error(f"Failed to fetch video description: {e}")
        raise HTTPException(status_code=500, detail="Unable to retrieve video description.")

def build_gpt_prompt(description: str, topic: str) -> str:
    """Construct the prompt used for segment extraction."""
    return f"""
You're helping a student find the best part of a YouTube video to learn about: "{topic}".

Below is the video description. It may include timestamps, chapter titles, or just a general summary:

\"\"\"{description}\"\"\"

🎯 Your job:
- If there are **chapters** with timestamps, choose the one most relevant to the topic.
- If no chapters exist, or if none clearly match the topic, return the **whole video** from start to end.

Estimate the start and end times in **seconds**, based on the timestamps or a reasonable guess.

🧾 Respond in this JSON format:

{{
  "segment_title": "Light Reactions",       
  "start": 0,
  "end": 600
}}

If you're unable to estimate the duration, return 0 for the end value.
"""

def extract_segment_or_full(description: str, topic: str) -> Optional[dict]:
    """Use GPT-3.5 to analyze YouTube description and extract the best segment."""
    prompt = build_gpt_prompt(description, topic)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logging.error(f"GPT extraction failed: {e}")
        logging.debug(f"GPT raw output: {content if 'content' in locals() else 'N/A'}")
        return None

# === Main Endpoint ===

@router.post("/api/video-learning", response_model=VideoLearningResponse)
async def video_learning(data: VideoLearningRequest):
    """
    Given a study topic, returns a YouTube video and timestamp segment (or full video)
    relevant to that topic.
    """
    topic = data.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")

    # Step 1: Search YouTube
    video = search_youtube_video(topic)
    video_id = video["id"]["videoId"]
    title = video["snippet"]["title"]

    # Step 2: Get Description
    description = get_video_description(video_id)

    # Step 3: Let GPT choose segment
    segment = extract_segment_or_full(description, topic)

    # Step 4: Return result
    if segment:
        return {
            "video_id": video_id,
            "title": title,
            "start": segment.get("start", 0),
            "end": segment.get("end", 0),
            "segment_title": segment.get("segment_title", None)
        }
    else:
        # Fallback to full video with no timestamps
        return {
            "video_id": video_id,
            "title": title,
            "start": 0,
            "end": 0,
            "segment_title": None
        }
