import argparse
import subprocess
import sys
import re
import logging
import math
from typing import List, Dict, Any
from PIL import Image
from io import BytesIO

logging.basicConfig(level=logging.INFO)

def get_video_duration(video_path: str) -> float:
    """
    Uses ffprobe to get total duration of the video in seconds.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Could not get video duration: {e}")
        return 0.0

def compute_average_color(video_path: str, start: float) -> tuple:
    """
    Extracts a frame at `start` seconds and computes the average RGB color.
    Returns (avg_r, avg_g, avg_b).
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss", str(start),  # seek to `start` seconds
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-f", "image2",
        "pipe:1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
        image = Image.open(BytesIO(result.stdout))
        image = image.convert("RGB")  # Ensure it's in RGB mode
        
        # Compute average color
        pixels = list(image.getdata())
        avg_r = sum(p[0] for p in pixels) / len(pixels)
        avg_g = sum(p[1] for p in pixels) / len(pixels)
        avg_b = sum(p[2] for p in pixels) / len(pixels)

        return (avg_r, avg_g, avg_b)
    except Exception as e:
        logging.error(f"Error extracting frame: {e}")
        # Fallback color if extraction fails
        return (0, 0, 0)

def color_distance(c1: tuple, c2: tuple) -> float:
    """
    Computes the Euclidean distance between two RGB colors.
    """
    return math.sqrt(
        (c1[0] - c2[0])**2 +
        (c1[1] - c2[1])**2 +
        (c1[2] - c2[2])**2
    )

def detect_scenes_and_assign_ids(
    input_video: str,
    threshold: float = 0.2,           # Lower => more scene changes
    similarity_threshold: float = 5.0 # Lower => more distinct scene IDs
) -> List[Dict[str, Any]]:
    """
    Detect scenes using ffmpeg's scene detection and assign IDs based on visual similarity.

    Returns a list of dicts like:
        [
            {
                "start": float,
                "end": float,
                "scene_id": int
            },
            ...
        ]
    """

    # 1) Use ffmpeg to detect scene changes
    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", input_video,
        "-filter_complex", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    
    logging.info(f"Running FFmpeg scene detection on {input_video} (threshold={threshold})...")
    try:
        # We capture stdout & stderr because showinfo logs to stderr
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error in scene detection: {e.stderr}\n")
        return []

    # 2) Parse timestamps from ffmpeg output: look for "time:xxx"
    timestamps = []
    combined_output = result.stdout + result.stderr
    for line in combined_output.splitlines():
        match = re.search(r"time:([\d\.]+)", line)
        if match:
            timestamps.append(float(match.group(1)))

    # Sort them in ascending order
    timestamps.sort()

    # Ensure 0.0 is the first scene boundary
    if not timestamps or (timestamps[0] > 0.0):
        timestamps.insert(0, 0.0)

    # 3) Append total duration as a final boundary
    total_duration = get_video_duration(input_video)
    if total_duration <= 0:
        logging.warning("Video duration is zero or couldn't be determined.")
        return []

    # If last timestamp is well before end, add the end
    if timestamps[-1] < (total_duration - 0.05):
        timestamps.append(total_duration)

    # If we only have 1 or 2 timestamps, might mean no real changes
    if len(timestamps) < 2:
        # The entire video is one scene
        return [{
            "start": 0.0,
            "end": total_duration,
            "scene_id": 0
        }]

    # 4) Build scene segments
    scenes = []
    scene_signatures = []  # (scene_id, avg_color)
    next_scene_id = 0

    for i in range(len(timestamps) - 1):
        start_time = timestamps[i]
        end_time   = timestamps[i + 1]

        # Midpoint to sample a representative frame
        midpoint = (start_time + end_time) / 2.0
        avg_color = compute_average_color(input_video, midpoint)

        # Compare with existing scene signatures
        assigned_id = None
        for scene_id, signature in scene_signatures:
            if color_distance(avg_color, signature) < similarity_threshold:
                assigned_id = scene_id
                break

        # If no match, create a new scene ID
        if assigned_id is None:
            assigned_id = next_scene_id
            scene_signatures.append((assigned_id, avg_color))
            next_scene_id += 1

        scenes.append({
            "start": start_time,
            "end":   end_time,
            "scene_id": assigned_id
        })

    return scenes


def clip_creation(args):
    import video_utils
    import os
    
    os.makedirs(args.save_folder, exist_ok=True)
    base_filename = os.path.basename(args.driving_path)

    # Scene detection
    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(
        args.driving_path,
        threshold=args.scene_threshold,
        similarity_threshold=args.scene_similarity_threshold
    )

    # ▼▼ ADDING A DEBUG PRINT FOR #unique scene IDs ▼▼
    unique_ids = set(s["scene_id"] for s in scenes)
    print(f"Detected {len(scenes)} scene segments in total.")
    print(f"[DEBUG] UNIQUE SCENE IDs = {sorted(unique_ids)} (TOTAL: {len(unique_ids)})")

    # Also, show each segment's ID/start/end
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        print(f"[DEBUG] scene_index={i} => scene_id={sid}, start={start}, end={end}")

    # Extract sub-videos for each scene
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)
    temp_driving_dir = os.path.join(args.save_folder, "temp_source")
    os.makedirs(temp_source_dir, exist_ok=True)

    scene_video_paths = []  # list[(scene_id, scene_video_path), ...]
    scene_ff_paths = []
    used_ids = set()
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        scene_vid_path = os.path.join(temp_driving_dir, f"scene_{i}.mp4")

        # Extract subvideo for [start, end]
        video_utils.extract_subvideo(args.driving_path, scene_vid_path, start, end)
        scene_video_paths.append((sid, scene_vid_path))

        # If first occurrence of scene_id, optionally extract first frame
        if sid not in used_ids:
            used_ids.add(sid)
            source_png = os.path.join(temp_source_dir, f"ff_{sid}.png")
            video_utils.extract_first_frame(scene_vid_path, source_png)
            scene_ff_paths.append(source_png)
            # (You commented this out, so we won't call it.)
    
    return scene_ff_paths


def get_scene_video_paths(args):
    """
    Looks inside args.save_folder/temp_driving/ for files named 'scene_<number>_id_<scene_id>.mp4'
    and returns a list of tuples (scene_id, full_path), sorted by <number> ascending.
    """
    import re
    import os
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    scene_paths = []

    # Regex pattern to extract (scene_number, scene_id)
    pattern = re.compile(r"scene_(\d+)_id_(\d+)\.mp4")

    for filename in os.listdir(temp_driving_dir):
        match = pattern.match(filename)
        if match:
            scene_number = int(match.group(1))  # Extract scene number
            scene_id = int(match.group(2))      # Extract scene ID
            full_path = os.path.join(temp_driving_dir, filename)
            
            scene_paths.append((scene_number, scene_id, full_path))

    # Sort by scene number (ascending)
    scene_paths.sort(key=lambda x: x[0])

    # Return only (scene_id, full_path)
    return [(scene_id, full_path) for _, scene_id, full_path in scene_paths]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scene changes in a video.")
    parser.add_argument("--vid_path", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--threshold", type=float, default=0.2,
                        help="FFmpeg scene detection threshold (lower => more sensitive).")
    parser.add_argument("--similarity_threshold", type=float, default=5.0,
                        help="Color-similarity threshold for merging scene IDs.")
    args = parser.parse_args()

    results = detect_scenes_and_assign_ids(
        input_video=args.vid_path,
        threshold=args.threshold,
        similarity_threshold=args.similarity_threshold
    )
    print(results)