import cv2
import os
import os.path as osp
import numpy as np
import tyro
import subprocess
from .src.config.argument_config import ArgumentConfig
from .src.config.inference_config import InferenceConfig
from .src.config.crop_config import CropConfig
from .src.live_portrait_pipeline import LivePortraitPipeline
from PIL import Image
# from AnonHead.segment_multiple import Segment
from .inference import main as motion_lp
from .utils.video_utils import merge_segments
import re
from tqdm import tqdm


def main(
    scenes,
    file_name = 'final',
    stitch = True,
    save_folder = './results'
    ):
    
    # Gather all scene videos in 'temp_driving/'
    processed_segments = []
    processed_segments_concat = [] 

    for (scene_number, scene_id, scene_video_path, source_image) in tqdm(scenes, desc="Processing Scenes"):
        print(f"[INFO] Processing scene_number={scene_number}, scene_id={scene_id}")
        print(f"       => {scene_video_path}")

        if "no_face" in source_image:
            processed_segments.append(scene_video_path)
            processed_segments_concat.append(scene_video_path)
            continue

        folder_name = 'temp_driving'
        processed_folder = os.path.join(save_folder, folder_name)
        os.makedirs(processed_folder, exist_ok=True)

        print(f"[INFO] -> Running Live Portrait on (source={source_image}, driving={scene_video_path})")

        args_lp = ArgumentConfig(
            source=source_image,
            driving=scene_video_path,
            flag_stitching = stitch,
            output_dir=processed_folder  # pass the folder, not a file
        )

        motion_lp(args_lp)

        final_scene_out = os.path.join(processed_folder, f"{file_name}_source_{scene_id}--{file_name}_scene_{scene_number}_id_{scene_id}.mp4")
        final_scene_out_concat = os.path.join(processed_folder, f"{file_name}_source_{scene_id}--{file_name}_scene_{scene_number}_id_{scene_id}_concat.mp4")
        if os.path.isfile(final_scene_out):
            processed_segments.append(final_scene_out)
        else:
            print(f"[WARN] {final_scene_out} not found. The pipeline may use a different name.")
        if os.path.isfile(final_scene_out_concat):
            processed_segments_concat.append(final_scene_out_concat)
        else:
            print(f"[WARN] {final_scene_out_concat} not found. The pipeline may use a different name.")

    if processed_segments:
        final_merged = os.path.join(save_folder, f"{file_name}_output_lp.mp4")
        final_merged_concat = os.path.join(save_folder, f"{file_name}_output_lp_concat.mp4")
        print(f"[INFO] Merging {len(processed_segments)} processed segments into {final_merged}")
        merge_segments(processed_segments, final_merged)
        merge_segments(processed_segments_concat, final_merged_concat)
        print(f"[INFO] Done! Final merged video: {final_merged}")
    else:
        print("[WARN] No processed segments to merge.")
        
        
