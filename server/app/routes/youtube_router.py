from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict
from services.youtube_service import YouTubeVideoService, ProcessedVideo, VideoInfo, CaptionSegment
from services.youtube_service import MongoDBEmbeddingSaver
from dataclasses import asdict
import asyncio
from datetime import datetime

router = APIRouter(
    prefix="/youtube",
    tags=["youtube"],
    responses={404: {"description": "Not found"}},
)

youtube_service = YouTubeVideoService()
mongodb_saver = MongoDBEmbeddingSaver()

@router.get("/video/info")
async def get_video_info(url: str) -> VideoInfo:
    """Get comprehensive video information"""
    try:
        video_info = await youtube_service.get_video_info(url)
        return video_info
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/video/captions")
async def get_video_captions(
    url: str,
    languages: Optional[List[str]] = Query(['en']),
    format_type: str = Query("json", regex="^(json|txt|srt|vtt)$"),
    translate_to: Optional[str] = None,
    prefer_manual: bool = True
) -> Dict:
    """Get video captions in specified format"""
    try:
        result = await youtube_service.get_captions(
            url=url,
            languages=languages,
            format_type=format_type,
            translate_to=translate_to,
            prefer_manual=prefer_manual
        )
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/video/captions/available")
async def get_available_captions(url: str) -> Dict:
    """Get list of all available caption tracks for a video"""
    try:
        result = await youtube_service.get_all_available_captions(url)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/video/process")
async def process_video(
    url: str,
    languages: List[str] = Query(['en']),
    embed_individual_segments: bool = True,
    save_to_db: bool = True
) -> Dict:
    """Process video with embeddings and optionally save to database"""
    try:
        processed_video, save_result = await youtube_service.process_video_with_embeddings(
            url=url,
            languages=languages,
            embed_individual_segments=embed_individual_segments,
            save_to_db=save_to_db
        )
        
        response = {
            "video_info": asdict(processed_video.video_info),
            "processing_results": {
                "captions_text_length": len(processed_video.captions_text),
                "caption_segments_count": len(processed_video.caption_segments),
                "metadata_embedding_generated": bool(processed_video.metadata_embedding),
                "full_text_embedding_generated": bool(processed_video.full_text_embedding)
            }
        }
        
        if save_to_db and save_result:
            response["save_result"] = save_result
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_videos(
    query: str,
    limit: int = Query(5, ge=1, le=20)
) -> List[Dict]:
    """Search for similar videos using vector search"""
    try:
        results = await youtube_service.search_videos(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/video/{video_id}")
async def get_video(video_id: str) -> Dict:
    """Get video data from MongoDB by video ID"""
    try:
        video_data = await get_video_by_id(video_id)
        if not video_data:
            raise HTTPException(status_code=404, detail="Video not found")
        return video_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/content")
async def search_videos_by_content(
    query: str,
    limit: int = Query(5, ge=1, le=20)
) -> List[Dict]:
    """Search videos by content similarity"""
    try:
        results = await search_videos_by_content(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def shutdown():
    """Clean up resources on shutdown"""
    youtube_service.close()
    mongodb_saver.close()