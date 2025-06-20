import os
import ffmpeg
import yt_dlp
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from urllib.parse import urlparse, parse_qs


def download_youtube_video(url, output_dir="downloads"):
    """
    Download YouTube video and return the path to the downloaded file.
    
    Args:
        url (str): YouTube video URL
        output_dir (str): Directory to save the downloaded video
    
    Returns:
        str: Path to the downloaded video file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure yt-dlp options
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Prefer mp4, fallback to best quality
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,  # Avoid special characters in filename
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'video')
            
            print(f"[*] Downloading: {video_title}")
            
            # Download the video
            ydl.download([url])
            
            # Get the actual filename
            filename = ydl.prepare_filename(info)
            
            # Handle potential extension changes
            if not os.path.exists(filename):
                # Try common extensions if the exact filename doesn't exist
                base_name = os.path.splitext(filename)[0]
                for ext in ['.mp4', '.webm', '.mkv']:
                    potential_file = base_name + ext
                    if os.path.exists(potential_file):
                        filename = potential_file
                        break
            
            if os.path.exists(filename):
                print(f"[*] Download completed: {filename}")
                return filename
            else:
                raise FileNotFoundError("Downloaded file not found")
                
    except Exception as e:
        print(f"[!] Error downloading video: {str(e)}")
        raise


def detect_scenes(video_path, threshold=15.0, min_scene_len=15):
    """
    Detect scene changes in a video file.
    
    Args:
        video_path (str): Path to the video file
        threshold (float): Sensitivity threshold for scene detection
        min_scene_len (int): Minimum scene length in frames
    
    Returns:
        list: List of scene timecodes
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()


def extract_scene_frames_ffmpeg(video_path, scenes, output_dir="scene_frames"):
    """
    Extract frames at scene change points using FFmpeg.
    
    Args:
        video_path (str): Path to the video file
        scenes (list): List of scene timecodes
        output_dir (str): Directory to save extracted frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (start_time, _) in enumerate(scenes):
        timestamp = start_time.get_seconds()
        output_file = os.path.join(output_dir, f"scene_{i + 1:03d}.jpg")
        
        try:
            (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output(output_file, vframes=1, q=2)  # Added quality setting
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            print(f"[!] Error extracting frame {i+1}: {str(e)}")
    
    print(f"[*] Extracted {len(scenes)} scene-change frames to: {output_dir}")


def extract_meaningful_frames(video_path, threshold=15.0, output_dir="scene_frames"):
    """
    Extract meaningful frames from a video based on scene detection.
    
    Args:
        video_path (str): Path to the video file
        threshold (float): Scene detection sensitivity
        output_dir (str): Directory to save extracted frames
    """
    print("[*] Detecting scenes...")
    scenes = detect_scenes(video_path, threshold=threshold)
    print(f"[*] Total scene cuts detected: {len(scenes)}")
    
    if len(scenes) < 5:
        print("[!] Very few scene cuts detected. Consider lowering threshold further.")
    elif len(scenes) > 100:
        print("[!] Many scene cuts detected. Consider raising threshold for fewer frames.")
    
    extract_scene_frames_ffmpeg(video_path, scenes, output_dir)


def process_youtube_video(url, threshold=15.0, download_dir="downloads", frames_dir="scene_frames"):
    """
    Complete pipeline: Download YouTube video and extract scene frames.
    
    Args:
        url (str): YouTube video URL
        threshold (float): Scene detection sensitivity
        download_dir (str): Directory for downloaded videos
        frames_dir (str): Directory for extracted frames
    
    Returns:
        str: Path to the downloaded video file
    """
    try:
        # Download the video
        video_path = download_youtube_video(url, download_dir)
        
        # Extract meaningful frames
        extract_meaningful_frames(video_path, threshold, frames_dir)
        
        return video_path
        
    except Exception as e:
        print(f"[!] Pipeline failed: {str(e)}")
        raise


def is_youtube_url(url):
    """
    Check if the provided URL is a YouTube URL.
    
    Args:
        url (str): URL to check
    
    Returns:
        bool: True if it's a YouTube URL
    """
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    parsed_url = urlparse(url)
    return parsed_url.netloc in youtube_domains


# Example usage:
if __name__ == "__main__":
    # Option 1: Process YouTube URL directly
    youtube_url = "https://www.youtube.com/watch?v=IgiIkW17ckM"
    
    # Option 2: Process local video file
    local_video_path = "downloaded_video.mp4"
    
    # Check if input is a URL or local file
    video_input = input("Enter YouTube URL or local video path: ")
    
    if is_youtube_url(video_input):
        print("[*] YouTube URL detected. Starting download and processing...")
        try:
            video_path = process_youtube_video(
                url=video_input,
                threshold=15.0,
                download_dir="downloads",
                frames_dir="scene_frames"
            )
            print(f"[*] Processing completed. Video saved at: {video_path}")
        except Exception as e:
            print(f"[!] Failed to process YouTube video: {str(e)}")
    else:
        if os.path.exists(video_input):
            print("[*] Local video file detected. Processing...")
            extract_meaningful_frames(video_input, threshold=15.0)
        else:
            print("[!] File not found. Please check the path or provide a valid YouTube URL.")