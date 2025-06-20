import os
import ffmpeg
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path, threshold=15.0, min_scene_len=15):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()


def extract_scene_frames_ffmpeg(video_path, scenes, output_dir="scene_frames"):
    os.makedirs(output_dir, exist_ok=True)
    for i, (start_time, _) in enumerate(scenes):
        timestamp = start_time.get_seconds()
        output_file = os.path.join(output_dir, f"scene_{i + 1:03d}.jpg")
        (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_file, vframes=1)
            .run(quiet=True, overwrite_output=True)
        )
    print(f"Extracted {len(scenes)} scene-change frames to: {output_dir}")


def extract_meaningful_frames(video_path, threshold=15.0):
    print("[*] Detecting scenes with lower threshold...")
    scenes = detect_scenes(video_path, threshold=threshold)
    print(f"[*] Total scene cuts detected: {len(scenes)}")

    if len(scenes) < 5:
        print("[!] Very few scene cuts detected. Consider lowering threshold further.")

    extract_scene_frames_ffmpeg(video_path, scenes)


# Example usage:
if __name__ == "__main__":
    video_path = "downloaded_video.mp4"
    extract_meaningful_frames(video_path, threshold=15.0)