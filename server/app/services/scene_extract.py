import os
import json
import ffmpeg
import yt_dlp
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from urllib.parse import urlparse
import google.generativeai as genai
from PIL import Image
from datetime import datetime
import time
import logging
from typing import Dict, List, Optional, Any
from collections import deque
import threading


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def download_youtube_video(url: str, output_dir: str = "downloads") -> str:
    """Download YouTube video completely and return the path."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Force single video download, not playlist
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
        'noplaylist': True,
        'extract_flat': False,
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("📥 Getting video info...")
            
            # Clean URL to ensure single video
            if 'list=' in url:
                url = url.split('&list=')[0].split('?list=')[0]
                logger.info(f"🔧 Cleaned URL to single video: {url}")
            
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'video')
            duration = info.get('duration', 0)
            
            logger.info(f"📺 Video: {video_title}")
            if duration:
                logger.info(f"⏱️  Duration: {duration//60}:{duration%60:02d}")
            logger.info("🔄 Starting single video download...")
            
            # Download the video
            ydl.download([url])
            filename = ydl.prepare_filename(info)
            
            # Handle potential extension changes
            if not os.path.exists(filename):
                base_name = os.path.splitext(filename)[0]
                for ext in ['.mp4', '.webm', '.mkv']:
                    potential_file = base_name + ext
                    if os.path.exists(potential_file):
                        filename = potential_file
                        break
            
            if os.path.exists(filename):
                logger.info(f"✅ Download completed: {os.path.basename(filename)}")
                return filename
            else:
                raise FileNotFoundError("Downloaded file not found")
                
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        raise


def detect_scenes(video_path: str, threshold: float = 15.0, min_scene_len: int = 15) -> List:
    """Detect scene changes in the downloaded video."""
    logger.info("🔍 Analyzing video for scene changes...")
    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
        scene_manager.detect_scenes(video)
        scenes = scene_manager.get_scene_list()
        logger.info(f"📊 Found {len(scenes)} scene transitions")
        return scenes
    except Exception as e:
        logger.error(f"Error detecting scenes: {e}")
        raise


class RateLimiter:
    """Rate limiter for API calls."""
    def __init__(self, max_calls: int = 12, time_window: int = 60):
        self.max_calls = max_calls  # Conservative limit (12 instead of 15)
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit."""
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and now - self.calls[0] > self.time_window:
                self.calls.popleft()
            
            # If we're at the limit, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0]) + 1
                if sleep_time > 0:
                    logger.info(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    # Clean up after sleeping
                    now = time.time()
                    while self.calls and now - self.calls[0] > self.time_window:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)


def setup_gemini_api(api_key: str):
    """Setup Gemini API for embeddings."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Test the API connection
        test_response = model.generate_content("Test connection")
        logger.info("✅ Gemini API connection successful")
        
        # Create rate limiter
        rate_limiter = RateLimiter(max_calls=12, time_window=60)  # Conservative limit
        
        return model, rate_limiter
    except Exception as e:
        logger.error(f"❌ Gemini API setup failed: {e}")
        raise


def extract_frame_with_retry(video_path: str, timestamp: float, output_file: str, max_retries: int = 3) -> bool:
    """Extract frame with retry logic and better error handling."""
    for attempt in range(max_retries):
        try:
            # Check if video file exists and is accessible
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get video info first to validate timestamp
            try:
                probe = ffmpeg.probe(video_path)
                duration = float(probe['streams'][0]['duration'])
                if timestamp >= duration:
                    logger.warning(f"Timestamp {timestamp}s exceeds video duration {duration}s")
                    timestamp = duration - 1  # Use last second of video
            except Exception as probe_error:
                logger.warning(f"Could not probe video duration: {probe_error}")
            
            # Extract frame
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(output_file, vframes=1, q=2, loglevel='error')
                .run(quiet=True, overwrite_output=True)
            )
            
            # Verify the frame was created and is valid
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                try:
                    # Test if image can be opened
                    with Image.open(output_file) as img:
                        img.verify()
                    return True
                except Exception as img_error:
                    logger.warning(f"Invalid image created on attempt {attempt + 1}: {img_error}")
                    if os.path.exists(output_file):
                        os.remove(output_file)
            
        except Exception as e:
            logger.warning(f"Frame extraction attempt {attempt + 1} failed: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    return False


def create_frame_and_embedding(model, rate_limiter, video_path: str, timestamp: float, frame_number: int, 
                             output_dir: str, max_retries: int = 3) -> Dict[str, Any]:
    """Extract frame and create embedding with improved error handling and rate limiting."""
    output_file = os.path.join(output_dir, f"scene_{frame_number:03d}.jpg")
    
    try:
        # Extract frame with retry logic
        if not extract_frame_with_retry(video_path, timestamp, output_file, max_retries):
            return {
                "frame_number": frame_number,
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                "frame_path": output_file,
                "description": "Failed to extract frame",
                "embedding": None,
                "status": "frame_extraction_failed"
            }
        
        # Create embedding if model is available
        if model and rate_limiter:
            try:
                image = Image.open(output_file)
                
                # Generate description with rate limiting and retry logic
                description = None
                for attempt in range(max_retries):
                    try:
                        # Wait for rate limit if needed
                        rate_limiter.wait_if_needed()
                        
                        response = model.generate_content([
                            "Describe this video frame in 1-2 sentences focusing on main elements, colors, and key objects or people visible.",
                            image
                        ])
                        
                        description = response.text.strip()
                        if description:
                            break
                            
                    except Exception as desc_error:
                        error_msg = str(desc_error)
                        if "429" in error_msg or "quota" in error_msg.lower():
                            logger.warning(f"Rate limit hit on attempt {attempt + 1}, waiting longer...")
                            time.sleep(15)  # Wait 15 seconds for rate limit
                        else:
                            logger.warning(f"Description generation attempt {attempt + 1} failed: {desc_error}")
                            if attempt < max_retries - 1:
                                time.sleep(2)
                        
                        if attempt == max_retries - 1:
                            description = "Description generation failed due to API limits"
                
                # Generate embedding from description
                if description and "failed" not in description.lower():
                    try:
                        # Rate limit for embedding as well
                        rate_limiter.wait_if_needed()
                        
                        embedding_response = genai.embed_content(
                            model="models/text-embedding-004",
                            content=description,
                            task_type="semantic_similarity"
                        )
                        
                        return {
                            "frame_number": frame_number,
                            "timestamp_seconds": round(timestamp, 2),
                            "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                            "frame_path": output_file,
                            "description": description,
                            "embedding": embedding_response['embedding'],
                            "status": "success"
                        }
                        
                    except Exception as embed_error:
                        embed_error_msg = str(embed_error)
                        if "429" in embed_error_msg or "quota" in embed_error_msg.lower():
                            logger.error(f"Rate limit hit during embedding: {embed_error}")
                            return {
                                "frame_number": frame_number,
                                "timestamp_seconds": round(timestamp, 2),
                                "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                                "frame_path": output_file,
                                "description": description,
                                "embedding": None,
                                "status": "rate_limited"
                            }
                        else:
                            logger.error(f"Embedding generation failed: {embed_error}")
                            return {
                                "frame_number": frame_number,
                                "timestamp_seconds": round(timestamp, 2),
                                "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                                "frame_path": output_file,
                                "description": description,
                                "embedding": None,
                                "status": "embedding_failed"
                            }
                else:
                    return {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(timestamp, 2),
                        "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        "frame_path": output_file,
                        "description": description or "No description generated",
                        "embedding": None,
                        "status": "description_failed"
                    }
                    
            except Exception as image_error:
                logger.error(f"Image processing failed: {image_error}")
                return {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(timestamp, 2),
                    "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                    "frame_path": output_file,
                    "description": f"Image processing error: {str(image_error)}",
                    "embedding": None,
                    "status": "image_processing_failed"
                }
        else:
            return {
                "frame_number": frame_number,
                "timestamp_seconds": round(timestamp, 2),
                "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                "frame_path": output_file,
                "description": "No API key provided",
                "embedding": None,
                "status": "no_api"
            }
            
    except Exception as e:
        logger.error(f"Unexpected error processing frame {frame_number}: {e}")
        return {
            "frame_number": frame_number,
            "timestamp_seconds": round(timestamp, 2),
            "timestamp_formatted": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
            "frame_path": output_file,
            "description": f"Unexpected error: {str(e)}",
            "embedding": None,
            "status": "unexpected_error"
        }


def process_video_frames_with_embeddings(video_path: str, gemini_api_key: Optional[str] = None, 
                                       threshold: float = 15.0, output_dir: str = "scene_frames", 
                                       json_output: str = "frame_embeddings.json") -> Optional[Dict]:
    """Main processing: detect scenes, extract frames, and create embeddings simultaneously."""
    
    # Step 1: Validate video file
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None
    
    # Step 2: Detect all scenes first
    try:
        scenes = detect_scenes(video_path, threshold, min_scene_len=15)
    except Exception as e:
        logger.error(f"Scene detection failed: {e}")
        return None
    
    if len(scenes) < 5:
        logger.warning("⚠️  Few scenes detected. Consider lowering threshold.")
    elif len(scenes) > 50:
        logger.warning(f"⚠️  {len(scenes)} scenes detected. This will take time!")
        confirm = input("Continue? (y/n): ")
        if confirm.lower() != 'y':
            return None
    
    # Step 3: Setup output directory and Gemini
    os.makedirs(output_dir, exist_ok=True)
    
    model = None
    rate_limiter = None
    if gemini_api_key:
        try:
            logger.info("🤖 Initializing Gemini API...")
            model, rate_limiter = setup_gemini_api(gemini_api_key)
            logger.info("✅ Gemini ready for embeddings")
        except Exception as e:
            logger.error(f"Gemini setup failed: {e}")
            logger.info("⚠️  Continuing without embeddings")
    else:
        logger.info("⚠️  No API key - will extract frames only")
    
    # Step 4: Initialize JSON structure
    frame_data = {
        "metadata": {
            "video_path": video_path,
            "total_frames": len(scenes),
            "threshold": threshold,
            "processing_start": datetime.now().isoformat(),
            "embedding_model": "gemini-1.5-flash + text-embedding-004" if model else "none"
        },
        "frames": []
    }
    
    # Step 5: Process each frame
    logger.info(f"🎬 Processing {len(scenes)} frames with embeddings...")
    start_time = time.time()
    
    successful_frames = 0
    successful_embeddings = 0
    
    for i, (start_time_scene, _) in enumerate(scenes):
        timestamp = start_time_scene.get_seconds()
        progress = f"[{i+1:3d}/{len(scenes)}]"
        
        logger.info(f"{progress} {timestamp:6.1f}s")
        
        # Extract frame and create embedding
        frame_info = create_frame_and_embedding(
            model, rate_limiter, video_path, timestamp, i + 1, output_dir
        )
        
        # Update counters
        if frame_info["status"] in ["success", "no_api"]:
            successful_frames += 1
        if frame_info["status"] == "success":
            successful_embeddings += 1
        
        # Show status
        status_icons = {
            "success": "✅ frame + embedding",
            "no_api": "📁 frame only",
            "frame_extraction_failed": "❌ frame extraction failed",
            "embedding_failed": "⚠️ frame ok, embedding failed",
            "image_processing_failed": "❌ image processing failed",
            "rate_limited": "⏳ rate limited",
            "description_failed": "⚠️ description failed",
            "unexpected_error": "❌ unexpected error"
        }
        
        status_msg = status_icons.get(frame_info["status"], f"❌ {frame_info['status']}")
        logger.info(f"    -> {status_msg}")
        
        # Add to data
        frame_data["frames"].append(frame_info)
        
        # Save progress every 10 frames
        if (i + 1) % 10 == 0 or i == len(scenes) - 1:
            frame_data["metadata"]["last_updated"] = datetime.now().isoformat()
            frame_data["metadata"]["frames_completed"] = i + 1
            frame_data["metadata"]["successful_frames"] = successful_frames
            frame_data["metadata"]["successful_embeddings"] = successful_embeddings
            
            # Calculate time estimates
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(scenes) - i - 1)
            
            # Save progress
            try:
                with open(json_output, 'w', encoding='utf-8') as f:
                    json.dump(frame_data, f, indent=2, ensure_ascii=False)
                logger.info(f"    💾 Progress saved | ETA: {remaining:.0f}s")
            except Exception as save_error:
                logger.error(f"Failed to save progress: {save_error}")
    
    # Final save with completion stats
    total_time = time.time() - start_time
    
    frame_data["metadata"].update({
        "processing_completed": datetime.now().isoformat(),
        "total_time_seconds": round(total_time, 2),
        "successful_frames": successful_frames,
        "successful_embeddings": successful_embeddings,
        "frame_success_rate": round(successful_frames / len(scenes) * 100, 1),
        "embedding_success_rate": round(successful_embeddings / len(scenes) * 100, 1) if model else 0
    })
    
    try:
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(frame_data, f, indent=2, ensure_ascii=False)
    except Exception as save_error:
        logger.error(f"Failed to save final results: {save_error}")
    
    # Summary
    logger.info("🎉 Processing completed!")
    logger.info(f"📊 Total scenes: {len(scenes)}")
    logger.info(f"📁 Successful frames: {successful_frames}")
    logger.info(f"🤖 Successful embeddings: {successful_embeddings}")
    logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_time/len(scenes):.1f}s per frame)")
    logger.info(f"📁 Frames saved to: {output_dir}")
    logger.info(f"📄 JSON saved to: {json_output}")
    
    return frame_data


def clean_youtube_url(url: str) -> str:
    """Clean YouTube URL to ensure single video download."""
    if 'list=' in url:
        if '&list=' in url:
            url = url.split('&list=')[0]
        elif '?list=' in url:
            base_url = url.split('?list=')[0]
            if '&' in url.split('?list=')[1]:
                other_params = url.split('?list=')[1].split('&', 1)[1]
                url = base_url + '?' + other_params
            else:
                url = base_url
    
    logger.info(f"🔗 Using URL: {url}")
    return url


def is_youtube_url(url: str) -> bool:
    """Check if URL is from YouTube."""
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    parsed_url = urlparse(url)
    return parsed_url.netloc in youtube_domains


def main():
    """Main execution flow."""
    print("🎬 Video Frame Embeddings Generator")
    print("=" * 45)
    
    # Get inputs
    api_key = os.getenv('GOOGLE_API_KEY') or input("🔑 Google AI API key (or press Enter to skip): ")
    if not api_key:
        api_key = None
        print("⚠️  No API key provided - will extract frames only")
    else:
        print("💡 TIP: Free tier allows ~12 requests/minute. Processing will be automatically paced.")
    
    video_input = input("📹 Video URL or path: ").strip()
    if not video_input:
        video_input = "https://www.youtube.com/watch?v=j4q0hHVdHMk"  # Default from your script
    
    # Clean YouTube URL if needed
    if is_youtube_url(video_input):
        video_input = clean_youtube_url(video_input)
    
    # Optional settings
    print("\n⚙️  Settings (Enter for defaults):")
    threshold = input("Scene threshold (15.0): ") or "15.0"
    threshold = float(threshold)
    
    frames_dir = input("Frames directory (scene_frames): ") or "scene_frames"
    json_file = input("JSON output (frame_embeddings.json): ") or "frame_embeddings.json"
    
    # Rate limiting options
    if api_key:
        print("\n⏱️  Rate Limiting Options:")
        print("1. Auto-pace (recommended for free tier)")
        print("2. Frames only now, embeddings later")
        choice = input("Choose option (1/2, default: 1): ") or "1"
        
        if choice == "2":
            print("📁 Will extract frames only. Run script again later with same settings to add embeddings.")
            api_key = None
    
    try:
        logger.info("🚀 Starting process...")
        
        if is_youtube_url(video_input):
            logger.info("📺 YouTube video detected")
            
            # Step 1: Download video first
            video_path = download_youtube_video(video_input, "downloads")
            
            # Step 2: Process frames with embeddings
            frame_data = process_video_frames_with_embeddings(
                video_path=video_path,
                gemini_api_key=api_key,
                threshold=threshold,
                output_dir=frames_dir,
                json_output=json_file
            )
            
        else:
            if os.path.exists(video_input):
                logger.info("📁 Local video file detected")
                
                # Process local video directly
                frame_data = process_video_frames_with_embeddings(
                    video_path=video_input,
                    gemini_api_key=api_key,
                    threshold=threshold,
                    output_dir=frames_dir,
                    json_output=json_file
                )
            else:
                logger.error("❌ File not found!")
                return
        
        if frame_data:
            frame_success_rate = frame_data['metadata']['frame_success_rate']
            embedding_success_rate = frame_data['metadata']['embedding_success_rate']
            logger.info(f"✨ Frame success rate: {frame_success_rate}%")
            if api_key:
                logger.info(f"✨ Embedding success rate: {embedding_success_rate}%")
                if embedding_success_rate < 80:
                    logger.info("💡 TIP: Consider running embeddings in smaller batches or upgrading to paid API tier")
    
    except KeyboardInterrupt:
        logger.info("⏹️  Interrupted - partial results saved")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()