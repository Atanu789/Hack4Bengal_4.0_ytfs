# app/schemas/youtube.py
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any


class VideoRequest(BaseModel):
    url: str
    quality: Optional[str] = "highest"

    @validator("url")
    def validate_youtube_url(cls, v):
        if not any(domain in v.lower() for domain in ['youtube.com', 'youtu.be']):
            raise ValueError("Must be a valid YouTube URL")
        return v

    @validator("quality")
    def validate_quality(cls, v):
        if v and v not in ["highest", "lowest", "144p", "240p", "360p", "480p", "720p", "1080p", "1440p", "2160p"]:
            raise ValueError("Invalid quality option")
        return v


class VideoFormat(BaseModel):
    quality: Optional[str] = None
    resolution: Optional[str] = None
    ext: Optional[str] = None
    url: Optional[str] = None
    filesize: Optional[int] = None


class VideoInfoResponse(BaseModel):
    success: bool
    video_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    duration: Optional[int] = None
    view_count: Optional[int] = None
    author: Optional[str] = None
    upload_date: Optional[str] = None
    thumbnail_url: Optional[str] = None
    best_format: Optional[VideoFormat] = None
    all_formats: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class VideoFormatDetail(BaseModel):
    itag: int
    quality: str
    resolution: Optional[str] = None
    ext: str
    url: str
    filesize: Optional[int] = None
    fps: Optional[int] = None


class VideoFormatsResponse(BaseModel):
    success: bool
    video_id: str
    title: str
    total_formats: int
    video_formats: List[VideoFormatDetail]
    audio_formats: List[VideoFormatDetail]



class SubtitleSegment(BaseModel):
    """Individual subtitle segment with timestamp"""
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Subtitle text
    duration: float  # Duration in seconds

class SubtitleTrack(BaseModel):
    """Subtitle track information"""
    language: str
    language_name: str
    is_auto_generated: bool
    segments: List[SubtitleSegment]

class SubtitlesResponse(BaseModel):
    """Response model for subtitle requests"""
    success: bool
    video_id: str
    title: str
    language: str
    language_name: str
    is_auto_generated: bool
    format_type: str
    total_segments: int
    duration: float  # Total video duration
    subtitles: List[SubtitleSegment]
    raw_content: Optional[str] = None  # For SRT/VTT formats

class AvailableLanguagesResponse(BaseModel):
    """Response model for available subtitle languages"""
    success: bool
    video_id: str
    available_languages: List[Dict[str, Any]]
    total_languages: int