# video_utils.py

import os
import cv2
import numpy as np

################################################################################
# Video Trimming
################################################################################

def get_video_length_seconds(video_path):
    """
    Return the duration (in seconds) of a video using OpenCV.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        raise ValueError(f"Cannot read valid FPS from {video_path}")
    return frame_count / fps

def extract_subvideo(input_path, output_path, start_sec, end_sec):
    """
    Extract portion [start_sec, end_sec] from input video using OpenCV 
    and save to output_path in MP4 format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()

def split_video_by_seconds(video_path, segment_length, output_dir):
    """
    Split a video into multiple segments of fixed `segment_length` in seconds.
    The last segment may be shorter if the video duration is not a multiple of segment_length.

    Args:
        video_path (str): Input video file path.
        segment_length (float): Length of each segment (seconds).
        output_dir (str): Directory to store the resulting segments.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_length_seconds(video_path)

    start = 0
    segment_index = 0

    while start < duration:
        end = start + segment_length
        if end > duration:
            end = duration

        out_path = os.path.join(output_dir, f"segment_{segment_index}.mp4")
        extract_subvideo(video_path, out_path, start, end)

        start = end
        segment_index += 1

    print(f"Split {video_path} into {segment_index} segments in {output_dir}.")

def merge_segments(segments_list, merged_path):
    """
    Merge a list of .mp4 segment files (in chronological order) into a single .mp4 video.
    Uses OpenCV for concatenation.

    Args:
        segments_list (list of str): Sorted list of segment paths to merge.
        merged_path (str): File path for the merged output video.
    """
    if not segments_list:
        print("No segments found to merge.")
        return

    # Read info from the first segment
    cap0 = cv2.VideoCapture(segments_list[0])
    fps = cap0.get(cv2.CAP_PROP_FPS)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(merged_path, fourcc, fps, (width, height))

    for seg_path in segments_list:
        cap = cv2.VideoCapture(seg_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()

    out.release()
    #print(f"Merged segments saved as {merged_path}.")

def split_video_in_chunks(video_path, chunk_length_sec, output_dir):
    """
    Split a video into multiple "big chunks" (e.g., 1-minute chunks). Each chunk is stored as
    chunk_0.mp4, chunk_1.mp4, etc. This is helpful if you have a very long video and want to
    process it in larger slices.

    Args:
        video_path (str): Input video file path.
        chunk_length_sec (float): Duration of each chunk in seconds.
        output_dir (str): Directory to store chunked outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    duration = get_video_length_seconds(video_path)

    start = 0.0
    chunk_index = 0

    while start < duration:
        end = min(start + chunk_length_sec, duration)
        chunk_path = os.path.join(output_dir, f"chunk_{chunk_index}.mp4")
        extract_subvideo(video_path, chunk_path, start, end)
        chunk_index += 1
        start = end

    print(f"Split {video_path} into {chunk_index} chunks (length ~{chunk_length_sec}s) in {output_dir}.")
    
    
def extract_first_frame(video_path, output_image_path):
    """
    Extract the first frame from video_path and save it as an image to output_image_path.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_image_path, frame)
    cap.release()
    
def create_concat_video(scene_video_path, source_image_path, final_merged_path, final_merged_concat_path):
    # Load the source image
    source_img = cv2.imread(source_image_path)
    if source_img is None:
        print(f"[ERROR] Could not load source image: {source_image_path}")
        return

    # Open original and anonymized videos
    cap_orig = cv2.VideoCapture(scene_video_path)
    cap_anon = cv2.VideoCapture(final_merged_path)

    if not cap_orig.isOpened():
        print(f"[ERROR] Could not open original video: {scene_video_path}")
        return
    if not cap_anon.isOpened():
        print(f"[ERROR] Could not open anonymized video: {final_merged_path}")
        return

    # Get resolution and fps from original video
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_orig.get(cv2.CAP_PROP_FPS)

    # Resize the source image to match the video frame height
    source_img_resized = cv2.resize(source_img, (height, height))

    # Prepare output video writer
    concat_width = width * 2 + height  # original | source | anonymized
    out = cv2.VideoWriter(final_merged_concat_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (concat_width, height))

    while True:
        ret_orig, frame_orig = cap_orig.read()
        ret_anon, frame_anon = cap_anon.read()

        if not ret_orig or not ret_anon:
            break

        # Resize video frames
        frame_orig_resized = cv2.resize(frame_orig, (width, height))
        frame_anon_resized = cv2.resize(frame_anon, (width, height))

        # Concatenate frames: original | source image | anonymized
        concat_frame = np.concatenate((frame_orig_resized, source_img_resized, frame_anon_resized), axis=1)
        out.write(concat_frame)

    # Release everything
    cap_orig.release()
    cap_anon.release()
    out.release()
    print(f"[INFO] Saved side-by-side video to {final_merged_concat_path}")
    
    


