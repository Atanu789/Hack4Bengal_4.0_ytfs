from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from app.services.youtube_service import YouTubeVideoService, ProcessedVideo
from pydantic import BaseModel

router = APIRouter(
    prefix="/youtube",
    tags=["YouTube Video Processing"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Initialize the service
youtube_service = YouTubeVideoService()

class VideoInfoResponse(BaseModel):
    video_id: str
    title: str
    author: str
    description: str
    duration: Optional[int]
    view_count: Optional[int]
    publish_date: Optional[str]
    thumbnail_url: str
    available_captions: List[dict]
    tags: List[str]
    category: Optional[str]

class CaptionSegmentResponse(BaseModel):
    text: str
    start_time: float
    end_time: float
    duration: float
    formatted_start: str
    formatted_end: str

class ProcessedVideoResponse(BaseModel):
    video_info: VideoInfoResponse
    captions_text: str
    total_segments: int
    segments_sample: List[CaptionSegmentResponse]

class SearchResult(BaseModel):
    content: str
    metadata: dict
    similarity_score: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

@router.get("/video/info", response_model=VideoInfoResponse, responses={400: {"model": ErrorResponse}})
async def get_video_info(url: str):
    """
    Get comprehensive information about a YouTube video
    
    - **url**: YouTube video URL
    """
    try:
        video_info = await youtube_service.get_video_info(url)
        return {
            "video_id": video_info.video_id,
            "title": video_info.title,
            "author": video_info.author,
            "description": video_info.description,
            "duration": video_info.duration,
            "view_count": video_info.view_count,
            "publish_date": video_info.publish_date,
            "thumbnail_url": video_info.thumbnail_url,
            "available_captions": [
                {
                    "language_code": track.language_code,
                    "language_name": track.language_name,
                    "is_auto_generated": track.is_auto_generated,
                    "is_translatable": track.is_translatable
                }
                for track in video_info.available_captions
            ],
            "tags": video_info.tags,
            "category": video_info.category
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/captions", responses={200: {"content": {"text/plain": {}}}, 400: {"model": ErrorResponse}})
async def get_captions(
    url: str,
    languages: Optional[List[str]] = Query(['en'], description="Preferred languages in order of preference"),
    format_type: str = Query('json', description="Output format (json, txt, vtt, srt)"),
    translate_to: Optional[str] = Query(None, description="Language code to translate to")
):
    """
    Get captions/subtitles for a YouTube video in various formats
    
    - **url**: YouTube video URL
    - **languages**: Preferred languages (default: ['en'])
    - **format_type**: Output format (json, txt, vtt, srt)
    - **translate_to**: Language code to translate captions to
    """
    try:
        result = await youtube_service.get_captions(
            url,
            languages=languages,
            format_type=format_type,
            translate_to=translate_to
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        if format_type == "json":
            return result
        else:
            return JSONResponse(
                content=result["text"],
                media_type="text/plain"
            )
    except Exception as e:
        logger.error(f"Error getting captions: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/process", response_model=ProcessedVideoResponse, responses={400: {"model": ErrorResponse}})
async def process_video(
    url: str,
    languages: Optional[List[str]] = Query(['en'], description="Preferred languages for captions"),
    embed_segments: bool = Query(True, description="Whether to generate embeddings for individual segments"),
    save_to_db: bool = Query(True, description="Whether to save results to database")
):
    """
    Process a YouTube video with embeddings
    
    - **url**: YouTube video URL
    - **languages**: Preferred languages for captions (default: ['en'])
    - **embed_segments**: Generate embeddings for individual segments (default: True)
    - **save_to_db**: Save results to database (default: True)
    """
    try:
        processed_video, save_result = await youtube_service.process_video_with_embeddings(
            url,
            languages=languages,
            embed_individual_segments=embed_segments,
            save_to_db=save_to_db
        )
        
        # Prepare response with a sample of segments
        segments_sample = [
            {
                "text": seg.text,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "duration": seg.duration,
                "formatted_start": seg.formatted_start,
                "formatted_end": seg.formatted_end
            }
            for seg in processed_video.caption_segments[:5]  # Return first 5 segments as sample
        ]
        
        return {
            "video_info": {
                "video_id": processed_video.video_info.video_id,
                "title": processed_video.video_info.title,
                "author": processed_video.video_info.author,
                "description": processed_video.video_info.description,
                "duration": processed_video.video_info.duration,
                "view_count": processed_video.video_info.view_count,
                "publish_date": processed_video.video_info.publish_date,
                "thumbnail_url": processed_video.video_info.thumbnail_url,
                "tags": processed_video.video_info.tags,
                "category": processed_video.video_info.category
            },
            "captions_text": processed_video.captions_text,
            "total_segments": len(processed_video.caption_segments),
            "segments_sample": segments_sample,
            "save_result": save_result
        }
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/search", response_model=List[SearchResult], responses={400: {"model": ErrorResponse}})
async def search_videos(
    query: str,
    limit: int = Query(5, description="Number of results to return", ge=1, le=20)
):
    """
    Search for similar videos using vector search
    
    - **query**: Search query text
    - **limit**: Number of results to return (1-20)
    """
    try:
        results = await youtube_service.search_videos(query, limit)
        return results
    except Exception as e:
        logger.error(f"Error searching videos: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/available-captions", responses={400: {"model": ErrorResponse}})
async def get_available_captions(url: str):
    """
    Get all available caption tracks for a video
    
    - **url**: YouTube video URL
    """
    try:
        result = await youtube_service.get_all_available_captions(url)
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
        return result
    except Exception as e:
        logger.error(f"Error getting available captions: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down"""
    youtube_service.close()
    logger.info("YouTube service cleanup complete")