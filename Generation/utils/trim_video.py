import cv2
import argparse
import os

def trim_video(input_file, start_sec, end_sec, save_folder):
    """
    Trims a video from start_sec to end_sec and saves it in the specified folder.

    Parameters:
        input_file (str): Path to the input video.
        start_sec (float): Start time in seconds.
        end_sec (float): End time in seconds.
        save_folder (str): Folder to save the trimmed video.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {input_file}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert start and end seconds to frame numbers
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Ensure valid frame range
    if start_frame >= total_frames:
        print(f"Error: Start time {start_sec}s exceeds video length.")
        cap.release()
        return
    if end_frame > total_frames:
        print(f"Warning: End time {end_sec}s exceeds video length. Trimming to end of video.")
        end_frame = total_frames

    # Create the output filename
    base_filename = os.path.basename(input_file)
    output_file = os.path.join(save_folder, f"trimmed_{start_sec}_{end_sec}_{base_filename}")

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Seek to the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Trimming {input_file} from {start_sec}s to {end_sec}s...")

    current_frame = start_frame
    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Trimmed video saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim a video using OpenCV.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--start_sec", type=float, default= 0, help="Start time in seconds.")
    parser.add_argument("--end_sec", type=float, required=True, help="End time in seconds.")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save the trimmed video.")

    args = parser.parse_args()

    # Ensure save folder exists
    os.makedirs(args.save_folder, exist_ok=True)

    trim_video(args.input_file, args.start_sec, args.end_sec, args.save_folder)
